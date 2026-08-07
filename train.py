import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data.dataset_generator import generate_agricultural_dataset
from src.models.drought_model import train_drought_lstm, train_drought_xgboost, prepare_xgboost_features
from src.models.advanced_transformer import train_drought_transformer
from src.models.pinn_drought_model import train_pinn_drought_model
from src.models.multitask_agri_net import train_multitask_model
from src.models.ensemble_model import DroughtEnsembleMetaLearner
from src.models.hyperparameter_tuner import tune_drought_xgboost
from src.models.explainable_ai import SatelliteExplainableAI
from src.models.yield_model import train_yield_models

def run_pipeline():
    print("==========================================================================")
    print(" Satellite Data Yield & Drought Stress Monitoring System - Training Core ")
    print("==========================================================================")
    
    # 1. Dataset Generation / Loading
    if not os.path.exists('data/drought_timeseries.csv') or not os.path.exists('data/yield_dataset.csv'):
        df_ts, df_yield = generate_agricultural_dataset(num_fields=1200)
    else:
        print("Loading pre-generated agricultural datasets...")
        df_ts = pd.read_csv('data/drought_timeseries.csv')
        df_yield = pd.read_csv('data/yield_dataset.csv')
        
    # Field-level Train/Val Split (80% Train, 20% Val)
    unique_fields = df_yield['field_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_fields)
    split_idx = int(0.8 * len(unique_fields))
    train_field_ids = set(unique_fields[:split_idx])
    val_field_ids = set(unique_fields[split_idx:])
    
    train_ts = df_ts[df_ts['field_id'].isin(train_field_ids)]
    val_ts = df_ts[df_ts['field_id'].isin(val_field_ids)]
    
    train_yield = df_yield[df_yield['field_id'].isin(train_field_ids)]
    val_yield = df_yield[df_yield['field_id'].isin(val_field_ids)]
    
    print(f"Data Split -> Train Fields: {len(train_field_ids)} | Validation Fields: {len(val_field_ids)}")
    
    # 2. Train Base, PINN, and Multi-Task Models
    lstm_model, lstm_f1 = train_drought_lstm(train_ts, val_ts, epochs=15, batch_size=64)
    xgb_drought, xgb_f1 = train_drought_xgboost(train_ts, val_ts)
    transformer_model, tf_f1 = train_drought_transformer(train_ts, val_ts, epochs=15, batch_size=64)
    pinn_model, pinn_f1 = train_pinn_drought_model(train_ts, val_ts, epochs=15, batch_size=64)
    multitask_model, mt_f1, mt_r2 = train_multitask_model(train_ts, train_yield, val_ts, val_yield, epochs=15, batch_size=64)
    
    # 3. Fit Hybrid Stacking Ensemble Meta-Learner
    ensemble_builder = DroughtEnsembleMetaLearner()
    meta_learner, ensemble_f1 = ensemble_builder.fit_ensemble(train_ts, val_ts)
    
    # 4. Optuna Automated Hyperparameter Tuning
    try:
        best_params = tune_drought_xgboost(train_ts, val_ts, n_trials=8)
    except Exception as e:
        print(f"Optuna Tuning Note: {e}")
    
    # 5. Train Yield Estimation Models
    xgb_yield, rf_yield, yield_metrics = train_yield_models(train_yield, val_yield)
    
    # 6. Generate Explainable AI (SHAP XAI) Reports
    xai = SatelliteExplainableAI()
    xai.generate_shap_report(val_ts)
    
    # 7. Generate TÜBİTAK Project Evaluation Reports & Figures
    os.makedirs('reports', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Figure 1: Model Comparison Bar Chart (6-Model Benchmark)
    plt.figure(figsize=(11, 5.5))
    model_names = ['LSTM', 'XGBoost', 'Transformer', 'PINN (Physics)', 'Multi-Task Net', 'Hybrid Ensemble']
    f1_scores = [lstm_f1 * 100, xgb_f1 * 100, tf_f1 * 100, pinn_f1 * 100, mt_f1 * 100, ensemble_f1 * 100]
    colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#06b6d4', '#e11d48']
    
    bars = plt.bar(model_names, f1_scores, color=colors, width=0.55)
    plt.ylim(0, 100)
    plt.title('14-Gün Kuraklık Stresi Modelleri Başarım Karşılaştırması (Macro F1 %)', fontsize=12, fontweight='bold')
    plt.ylabel('Macro F1 Score (%)', fontsize=11)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"%{yval:.1f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('reports/drought_model_comparison.png', dpi=300)
    plt.close()
    
    # Figure 2: Drought Model Confusion Matrix
    plt.figure(figsize=(7, 5.5))
    X_val, y_val, _ = prepare_xgboost_features(val_ts)
    y_pred_xgb = xgb_drought.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_xgb)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=['Normal', 'Orta Stres', 'Şiddetli Kuraklık'],
                yticklabels=['Normal', 'Orta Stres', 'Şiddetli Kuraklık'])
    plt.title('14-Gün Kuraklık Stresi Erken Uyarı Modeli (Confusion Matrix)', fontsize=12, fontweight='bold')
    plt.xlabel('Tahmin Edilen Durum (Predicted)', fontsize=11)
    plt.ylabel('Gerçek Durum (Actual)', fontsize=11)
    plt.tight_layout()
    plt.savefig('reports/drought_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Figure 3: Yield Actual vs Predicted Plot
    plt.figure(figsize=(7, 5.5))
    X_yield_val = val_yield[[
        'soil_type', 'is_irrigated', 'peak_ndvi', 'avg_ndvi',
        'avg_ndwi', 'min_ndwi', 'min_soil_moisture', 'avg_soil_moisture',
        'total_precip_mm', 'avg_temp_c'
    ]]
    yield_preds = xgb_yield.predict(X_yield_val)
    
    plt.scatter(val_yield['yield_kg_per_da'], yield_preds, alpha=0.6, color='#2e7d32', edgecolors='k', s=40)
    plt.plot([100, 750], [100, 750], 'r--', label='Ideal 1:1 Match')
    plt.title(f'Rekolte Tahmin Modeli (Gerçek vs Tahmin)\nR² Score = {yield_metrics["r2"]:.3f}', fontsize=12, fontweight='bold')
    plt.xlabel('Gerçek Rekolte (kg/dönüm)', fontsize=11)
    plt.ylabel('Yapay Zeka Tahmini (kg/dönüm)', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/yield_actual_vs_predicted.png', dpi=300)
    plt.close()

    print("\n==========================================================================")
    print(" SUCCESS: Multi-Task Learning, Focal Loss & Full Pipeline Completed!")
    print(f" Reports & Charts generated in: {os.path.abspath('reports')}")
    print("==========================================================================")

if __name__ == "__main__":
    run_pipeline()
