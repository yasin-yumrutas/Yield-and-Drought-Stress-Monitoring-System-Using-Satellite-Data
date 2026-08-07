import os
import numpy as np
import pandas as pd
import torch
import joblib
from src.models.drought_model import DroughtLSTM

class CropHealthPredictor:
    """
    Inference Engine for Satellite Crop Health & Drought Stress Monitoring
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.lstm_model = None
        self.xgb_drought = None
        self.xgb_yield = None
        self._load_models()
        
    def _load_models(self):
        # 1. Load XGBoost Drought
        xgb_drought_path = os.path.join(self.models_dir, 'drought_xgboost.joblib')
        if os.path.exists(xgb_drought_path):
            self.xgb_drought = joblib.load(xgb_drought_path)
            
        # 2. Load PyTorch LSTM Drought
        lstm_path = os.path.join(self.models_dir, 'drought_lstm_best.pth')
        if os.path.exists(lstm_path):
            self.lstm_model = DroughtLSTM(input_dim=7, hidden_dim=64, num_layers=2, num_classes=3)
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
            self.lstm_model.eval()
            
        # 3. Load XGBoost Yield
        yield_path = os.path.join(self.models_dir, 'yield_xgboost.joblib')
        if os.path.exists(yield_path):
            self.xgb_yield = joblib.load(yield_path)

    def predict_field_health(self, time_series_df):
        """
        Predicts 14-day early warning stress risk and seasonal yield for a given field time series.
        """
        if self.xgb_drought is None or self.xgb_yield is None:
            raise FileNotFoundError("Trained models not found! Please run 'python train.py' first.")

        # Ensure correct ordering
        df_sorted = time_series_df.sort_values('step').copy()
        
        # Prepare rolling features for XGBoost Drought Model
        df_sorted['delta_ndvi_1'] = df_sorted['ndvi'].diff(1).fillna(0)
        df_sorted['delta_ndwi_1'] = df_sorted['ndwi'].diff(1).fillna(0)
        df_sorted['delta_soil_1'] = df_sorted['soil_moisture'].diff(1).fillna(0)
        df_sorted['rolling_mean_soil'] = df_sorted['soil_moisture'].rolling(3, min_periods=1).mean()
        df_sorted['rolling_mean_ndwi'] = df_sorted['ndwi'].rolling(3, min_periods=1).mean()
        
        feature_cols = [
            'ndvi', 'ndwi', 'temp_c', 'precip_mm', 'soil_moisture', 'soil_type', 'is_irrigated',
            'delta_ndvi_1', 'delta_ndwi_1', 'delta_soil_1', 'rolling_mean_soil', 'rolling_mean_ndwi'
        ]
        
        latest_features = df_sorted[feature_cols].iloc[[-1]]
        
        # Drought Prediction (Probabilities for 0: Normal, 1: Orta Stres, 2: Şiddetli Kuraklık)
        drought_probs = self.xgb_drought.predict_proba(latest_features)[0]
        predicted_class = int(np.argmax(drought_probs))
        
        class_labels = ['Normal (Sağlıklı)', 'Orta Su Stresi (Erken Uyarı)', 'Şiddetli Kuraklık Stresi']
        color_codes = ['#4CAF50', '#FF9800', '#F44336']  # Green, Orange, Red
        
        # Yield Forecast
        yield_input = pd.DataFrame([{
            'soil_type': df_sorted['soil_type'].iloc[-1],
            'is_irrigated': df_sorted['is_irrigated'].iloc[-1],
            'peak_ndvi': df_sorted['ndvi'].max(),
            'avg_ndvi': df_sorted['ndvi'].mean(),
            'avg_ndwi': df_sorted['ndwi'].mean(),
            'min_ndwi': df_sorted['ndwi'].min(),
            'min_soil_moisture': df_sorted['soil_moisture'].min(),
            'avg_soil_moisture': df_sorted['soil_moisture'].mean(),
            'total_precip_mm': df_sorted['precip_mm'].sum(),
            'avg_temp_c': df_sorted['temp_c'].mean()
        }])
        
        estimated_yield = float(self.xgb_yield.predict(yield_input)[0])
        
        result = {
            'latest_step': int(df_sorted['step'].iloc[-1]),
            'current_ndvi': float(df_sorted['ndvi'].iloc[-1]),
            'current_ndwi': float(df_sorted['ndwi'].iloc[-1]),
            'predicted_14d_stress_status': class_labels[predicted_class],
            'stress_level_code': predicted_class,
            'map_color_code': color_codes[predicted_class],
            'stress_probabilities': {
                'normal_prob': float(drought_probs[0]),
                'mild_stress_prob': float(drought_probs[1]),
                'severe_drought_prob': float(drought_probs[2])
            },
            'forecasted_yield_kg_per_da': round(estimated_yield, 1)
        }
        return result
