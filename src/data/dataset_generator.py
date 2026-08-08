import os
import numpy as np
import pandas as pd
from src.data.openmeteo_fetcher import OpenMeteoAgriFetcher

def generate_agricultural_dataset(num_fields=2400, steps_per_field=20, random_seed=42):
    """
    Generates multi-temporal satellite (Sentinel-2 NDVI, NDWI) and agro-climate time series.
    Uses continuous Crop Water Stress Index (CWSI) with realistic stochastic environmental noise
    to prevent artificial 99% deterministic probability saturation.
    """
    np.random.seed(random_seed)
    
    time_series_rows = []
    yield_rows = []

    print(f"Generating realistic satellite time-series dataset for {num_fields} agricultural fields...")

    for field_id in range(1, num_fields + 1):
        # Soil & Irrigation profile
        soil_type = np.random.choice([1, 2, 3], p=[0.25, 0.50, 0.25])  # 1: Sandy, 2: Loam, 3: Clay
        is_irrigated = np.random.choice([0, 1], p=[0.4, 0.6])
        
        # Drought Scenario Trigger
        scenario = np.random.choice(['healthy', 'mild_drought', 'severe_drought'], p=[0.45, 0.35, 0.20])
        
        # Base agronomic growth curve (bell-shaped NDVI trajectory over 100 days)
        steps = np.arange(steps_per_field)
        base_ndvi = 0.15 + 0.70 * np.exp(-((steps - 11) ** 2) / 25.0)
        base_ndwi = 0.10 + 0.40 * np.exp(-((steps - 11) ** 2) / 30.0)
        
        field_ndvis = []
        field_ndwis = []
        field_temps = []
        field_precip = []
        field_soil_m = []
        
        # Stress onset step
        stress_onset = np.random.randint(6, 12)
        
        for step in steps:
            days = step * 5
            
            # Base weather
            temp = 18.0 + 12.0 * np.sin(np.pi * step / steps_per_field) + np.random.normal(0, 2.0)
            precip = max(0.0, np.random.exponential(scale=6.0) if np.random.rand() > 0.6 else 0.0)
            
            # Soil moisture simulation with soil-dependent retention
            retention_mult = 0.8 if soil_type == 1 else (1.0 if soil_type == 2 else 1.2)
            soil_m = 25.0 + (precip * 0.8 * retention_mult) - (temp * 0.3) + np.random.normal(0, 2.5)
            if is_irrigated and step % 3 == 0:
                soil_m += 14.0
            
            soil_m = np.clip(soil_m, 5.0, 45.0)
            
            # Apply scenario stress impact with natural micro-variability
            ndvi = base_ndvi[step] + np.random.normal(0, 0.03)
            ndwi = base_ndwi[step] + np.random.normal(0, 0.03)
            
            if scenario == 'mild_drought' and step >= stress_onset:
                stress_factor = (step - stress_onset + 1) * 0.04
                ndvi -= stress_factor * 0.45 + np.random.normal(0, 0.015)
                ndwi -= stress_factor * 0.65 + np.random.normal(0, 0.02)
                soil_m -= stress_factor * 2.2
            elif scenario == 'severe_drought' and step >= stress_onset:
                stress_factor = (step - stress_onset + 1) * 0.08
                ndvi -= stress_factor * 0.75 + np.random.normal(0, 0.02)
                ndwi -= stress_factor * 1.05 + np.random.normal(0, 0.025)
                soil_m -= stress_factor * 3.8
                temp += 3.0
            
            ndvi = float(np.clip(ndvi, 0.05, 0.92))
            ndwi = float(np.clip(ndwi, -0.40, 0.55))
            soil_m = float(np.clip(soil_m, 4.0, 45.0))
            
            field_ndvis.append(ndvi)
            field_ndwis.append(ndwi)
            field_temps.append(temp)
            field_precip.append(precip)
            field_soil_m.append(soil_m)
            
        # PROBABILISTIC AGRONOMIC WATER STRESS INDEX (CWSI)
        # Replaces deterministic hard IF/THEN rules to reflect real-world natural class overlap
        for step in range(steps_per_field):
            future_step = min(step + 2, steps_per_field - 1)
            f_ndwi = field_ndwis[future_step]
            f_soil = field_soil_m[future_step]
            f_temp = field_temps[future_step]
            
            # Continuous Water Stress Index (0.0 to 1.0)
            cwsi = (0.45 * (0.30 - f_ndwi)) + (0.35 * (25.0 - f_soil) / 20.0) + (0.20 * (f_temp - 22.0) / 15.0)
            cwsi += np.random.normal(0, 0.08)  # Natural biological variability
            cwsi = np.clip(cwsi, 0.0, 1.0)
            
            # Probabilistic Soft Categorization
            if cwsi > 0.55:
                stress_target = 2  # Severe Stress
            elif cwsi > 0.30:
                stress_target = 1  # Mild Stress
            else:
                stress_target = 0  # Healthy
                
            time_series_rows.append({
                'field_id': field_id,
                'step': step,
                'days_from_sowing': step * 5,
                'ndvi': field_ndvis[step],
                'ndwi': field_ndwis[step],
                'temp_c': field_temps[step],
                'precip_mm': field_precip[step],
                'soil_moisture': field_soil_m[step],
                'soil_type': soil_type,
                'is_irrigated': is_irrigated,
                'stress_target_14d': stress_target
            })
            
        # Calculate seasonal yield ground truth (kg/dönüm)
        peak_ndvi = max(field_ndvis)
        avg_ndwi = np.mean(field_ndwis)
        min_soil_m = min(field_soil_m)
        total_precip = sum(field_precip)
        
        base_yield = 580.0
        yield_kg = base_yield * (peak_ndvi / 0.85) * (1.0 + 0.3 * avg_ndwi)
        if min_soil_m < 10.0:
            yield_kg *= 0.65
        elif min_soil_m < 15.0:
            yield_kg *= 0.82
            
        if not is_irrigated and total_precip < 50.0:
            yield_kg *= 0.78
            
        yield_kg += np.random.normal(0, 18.0)
        yield_kg = float(np.clip(yield_kg, 120.0, 750.0))
        
        yield_rows.append({
            'field_id': field_id,
            'soil_type': soil_type,
            'is_irrigated': is_irrigated,
            'peak_ndvi': peak_ndvi,
            'avg_ndvi': np.mean(field_ndvis),
            'avg_ndwi': avg_ndwi,
            'min_ndwi': min(field_ndwis),
            'min_soil_moisture': min_soil_m,
            'avg_soil_moisture': np.mean(field_soil_m),
            'total_precip_mm': total_precip,
            'avg_temp_c': np.mean(field_temps),
            'yield_kg_per_da': yield_kg
        })

    df_ts = pd.DataFrame(time_series_rows)
    df_yield = pd.DataFrame(yield_rows)
    
    os.makedirs('data', exist_ok=True)
    df_ts.to_csv('data/drought_timeseries.csv', index=False)
    df_yield.to_csv('data/yield_dataset.csv', index=False)
    
    print(f"Realistic Dataset successfully created! Time series samples: {len(df_ts)}, Field summary samples: {len(df_yield)}")
    return df_ts, df_yield

if __name__ == "__main__":
    generate_agricultural_dataset()
