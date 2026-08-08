import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.inference import CropHealthPredictor

def run_stress_test_validation():
    print("==========================================================================")
    print(" REAL-WORLD SAHA GERÇEKÇİLİK TESTİ & AYIKLAMA BENCHMARKI ")
    print("==========================================================================")
    
    predictor = CropHealthPredictor()
    
    # --------------------------------------------------------------------------
    # SAHA 1: SAĞLIKLI / DÜZENLİ SULANAN TARLA (Gaziantep Şahinbey Sulamalı Buğday)
    # High NDWI (> 0.20), High Soil Moisture (> 28%), Regular Rain/Irrigation
    # --------------------------------------------------------------------------
    healthy_steps = []
    for step in range(20):
        ndvi = 0.20 + 0.65 * np.exp(-((step - 10) ** 2) / 20.0) + np.random.normal(0, 0.01)
        ndwi = 0.15 + 0.35 * np.exp(-((step - 10) ** 2) / 25.0) + np.random.normal(0, 0.01)
        healthy_steps.append({
            'field_id': 101,
            'step': step,
            'days_from_sowing': step * 5,
            'ndvi': float(np.clip(ndvi, 0.15, 0.88)),
            'ndwi': float(np.clip(ndwi, 0.05, 0.50)),
            'temp_c': float(20.0 + 5.0 * np.sin(step / 3.0)),
            'precip_mm': float(12.0 if step % 3 == 0 else 2.0),
            'soil_moisture': float(28.0 + 5.0 * np.sin(step / 2.0)),
            'soil_type': 2,
            'is_irrigated': 1
        })
    df_healthy = pd.DataFrame(healthy_steps)
    res_healthy = predictor.predict_field_health(df_healthy)
    
    # --------------------------------------------------------------------------
    # SAHA 2: ORTA DERECEDE SU STRESİ (Gaziantep İslahiye Erken Uyarı Sahası)
    # Mild Water Deficit (NDWI ~0.08, Soil Moisture ~18%), Early Warning State
    # --------------------------------------------------------------------------
    mild_steps = []
    for step in range(20):
        ndvi = 0.20 + 0.58 * np.exp(-((step - 10) ** 2) / 22.0) - (0.008 * step)
        ndwi = 0.12 + 0.28 * np.exp(-((step - 10) ** 2) / 25.0) - (0.010 * step)
        mild_steps.append({
            'field_id': 102,
            'step': step,
            'days_from_sowing': step * 5,
            'ndvi': float(np.clip(ndvi, 0.12, 0.78)),
            'ndwi': float(np.clip(ndwi, 0.01, 0.38)),
            'temp_c': float(22.0 + step * 0.3),
            'precip_mm': float(4.0 if step % 4 == 0 else 0.0),
            'soil_moisture': float(max(15.5, 25.0 - step * 0.45)),
            'soil_type': 2,
            'is_irrigated': 0
        })
    df_mild = pd.DataFrame(mild_steps)
    res_mild = predictor.predict_field_health(df_mild)
    
    # --------------------------------------------------------------------------
    # SAHA 3: ŞİDDETLİ GERÇEK KURAKLIK TARLASI (Gaziantep Araban Aşırı Kuraklık)
    # Very Low NDWI (<-0.25), Low Soil Moisture (<8%), High Heat (>36°C), Zero Rain
    # --------------------------------------------------------------------------
    severe_steps = []
    for step in range(20):
        ndvi = max(0.08, 0.55 - step * 0.035)
        ndwi = max(-0.35, 0.10 - step * 0.04)
        severe_steps.append({
            'field_id': 103,
            'step': step,
            'days_from_sowing': step * 5,
            'ndvi': float(ndvi),
            'ndwi': float(ndwi),
            'temp_c': float(26.0 + step * 0.6),
            'precip_mm': 0.0,
            'soil_moisture': float(max(4.5, 20.0 - step * 1.0)),
            'soil_type': 1,
            'is_irrigated': 0
        })
    df_severe = pd.DataFrame(severe_steps)
    res_severe = predictor.predict_field_health(df_severe)
    
    # PRINT COMPARATIVE VALIDATION TABLE
    print("\n" + "="*85)
    print(f"{'TARLA SAHA TİPİ':<30} | {'TAHMİN DURUMU':<28} | {'SAĞLIKLI %':<10} | {'ORTA %':<8} | {'KURAK %':<8} | {'REKOLTE'}")
    print("="*85)
    
    h_p = res_healthy['stress_probabilities']
    m_p = res_mild['stress_probabilities']
    s_p = res_severe['stress_probabilities']
    
    print(f"{'TARLA A (Sağlıklı / Sulanan)':<30} | {res_healthy['predicted_14d_stress_status']:<28} | {h_p['normal_prob']*100:<9.1f}% | {h_p['mild_stress_prob']*100:<7.1f}% | {h_p['severe_drought_prob']*100:<7.1f}% | {res_healthy['forecasted_yield_kg_per_da']} kg")
    print(f"{'TARLA B (Orta Derece Stres)':<30} | {res_mild['predicted_14d_stress_status']:<28} | {m_p['normal_prob']*100:<9.1f}% | {m_p['mild_stress_prob']*100:<7.1f}% | {m_p['severe_drought_prob']*100:<7.1f}% | {res_mild['forecasted_yield_kg_per_da']} kg")
    print(f"{'TARLA C (Şiddetli Tam Kuraklık)':<30} | {res_severe['predicted_14d_stress_status']:<28} | {s_p['normal_prob']*100:<9.1f}% | {s_p['mild_stress_prob']*100:<7.1f}% | {s_p['severe_drought_prob']*100:<7.1f}% | {res_severe['forecasted_yield_kg_per_da']} kg")
    print("="*85)

if __name__ == "__main__":
    run_stress_test_validation()
