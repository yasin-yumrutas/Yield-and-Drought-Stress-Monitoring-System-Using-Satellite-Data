import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.drought_model import prepare_xgboost_features

class SatelliteExplainableAI:
    """
    Explainable AI (XAI) Engine using SHAP (SHapley Additive exPlanations)
    to explain AI satellite drought risk & yield predictions for TÜBİTAK presentation.
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.drought_model = None
        self.yield_model = None
        self._load_models()

    def _load_models(self):
        d_path = os.path.join(self.models_dir, 'drought_xgboost.joblib')
        y_path = os.path.join(self.models_dir, 'yield_xgboost.joblib')
        
        if os.path.exists(d_path):
            self.drought_model = joblib.load(d_path)
        if os.path.exists(y_path):
            self.yield_model = joblib.load(y_path)

    def generate_shap_report(self, val_df):
        """
        Generates feature attribution and SHAP value summary plots.
        """
        print("\n========================================================")
        print(" Generating Explainable AI (XAI) SHAP Feature Attribution ")
        print("========================================================")
        
        try:
            import shap
            
            X_val, _, feature_cols = prepare_xgboost_features(val_df)
            
            # TreeExplainer for Drought Classifier
            explainer_drought = shap.TreeExplainer(self.drought_model)
            shap_values_drought = explainer_drought.shap_values(X_val.sample(min(300, len(X_val)), random_state=42))
            
            os.makedirs('reports', exist_ok=True)
            
            # Figure: SHAP Summary Bar Chart
            plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values_drought, X_val.sample(min(300, len(X_val)), random_state=42),
                              feature_names=feature_cols,
                              class_names=['Normal', 'Orta Stres', 'Şiddetli Kuraklık'],
                              show=False)
            plt.title('Açıklanabilir Yapay Zeka (SHAP) - Kuraklık Karar Nedenleri', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig('reports/shap_drought_explanation.png', dpi=300)
            plt.close()
            
            # TreeExplainer for Yield Model
            feature_cols_yield = [
                'soil_type', 'is_irrigated', 'peak_ndvi', 'avg_ndvi',
                'avg_ndwi', 'min_ndwi', 'min_soil_moisture', 'avg_soil_moisture',
                'total_precip_mm', 'avg_temp_c'
            ]
            X_yield = val_df[feature_cols_yield]
            explainer_yield = shap.TreeExplainer(self.yield_model)
            shap_values_yield = explainer_yield.shap_values(X_yield)
            
            plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values_yield, X_yield, show=False)
            plt.title('Açıklanabilir Yapay Zeka (SHAP) - Rekolteye Etki Eden Etkenler', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig('reports/shap_yield_explanation.png', dpi=300)
            plt.close()
            
            print("SUCCESS: XAI SHAP plots exported to 'reports/shap_drought_explanation.png'")
            
        except Exception as e:
            print(f"SHAP Attribution Note: {e}")
            # Fallback to feature importance bar plot if SHAP is installing
            self._generate_fallback_attribution(val_df)

    def _generate_fallback_attribution(self, val_df):
        os.makedirs('reports', exist_ok=True)
        feature_cols_yield = [
            'soil_type', 'is_irrigated', 'peak_ndvi', 'avg_ndvi',
            'avg_ndwi', 'min_ndwi', 'min_soil_moisture', 'avg_soil_moisture',
            'total_precip_mm', 'avg_temp_c'
        ]
        importances = self.yield_model.feature_importances_
        
        plt.figure(figsize=(9, 5.5))
        sns.barplot(x=importances, y=feature_cols_yield, palette='viridis')
        plt.title('Açıklanabilir Yapay Zeka (XAI) - Rekolte Karar Nedenleri (Feature Attribution)', fontsize=12, fontweight='bold')
        plt.xlabel('Matematiksel Karar Ağırlığı (Attribution Score)', fontsize=11)
        plt.tight_layout()
        plt.savefig('reports/shap_yield_explanation.png', dpi=300)
        plt.close()
        print("Fallback XAI Feature Attribution chart saved to 'reports/shap_yield_explanation.png'")

if __name__ == "__main__":
    if os.path.exists('data/drought_timeseries.csv'):
        val_df = pd.read_csv('data/drought_timeseries.csv')
        xai = SatelliteExplainableAI()
        xai.generate_shap_report(val_df)
