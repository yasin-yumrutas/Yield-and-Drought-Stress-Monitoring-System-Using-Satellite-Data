import numpy as np
import pandas as pd

class TimeSeriesAgriAugmentor:
    """
    Time-Series Data Augmentor for Agricultural Satellite & Climate Trajectories.
    Simulates extreme weather, heatwaves, and cloud coverage noise to make models robust.
    """
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)

    def augment_field_series(self, df_field, num_augmented_copies=2):
        """
        Creates augmented variations of a single field's time series.
        """
        augmented_dfs = [df_field.copy()]
        base_id = df_field['field_id'].iloc[0]
        
        for i in range(1, num_augmented_copies + 1):
            df_aug = df_field.copy()
            df_aug['field_id'] = base_id * 1000 + i
            
            # 1. Jittering (Add subtle sensor noise to NDVI & NDWI)
            ndvi_noise = np.random.normal(0, 0.02, size=len(df_aug))
            ndwi_noise = np.random.normal(0, 0.02, size=len(df_aug))
            df_aug['ndvi'] = np.clip(df_aug['ndvi'] + ndvi_noise, 0.05, 0.90)
            df_aug['ndwi'] = np.clip(df_aug['ndwi'] + ndwi_noise, -0.35, 0.50)
            
            # 2. Temperature Heatwave Shift (Simulate unexpected summer heat spike)
            if np.random.rand() > 0.5:
                heat_spike = np.random.uniform(1.5, 4.0)
                df_aug['temp_c'] += heat_spike
                df_aug['soil_moisture'] = np.clip(df_aug['soil_moisture'] - (heat_spike * 0.8), 4.0, 45.0)
                
            augmented_dfs.append(df_aug)
            
        return pd.concat(augmented_dfs, ignore_index=True)
