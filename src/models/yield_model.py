import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

def train_yield_models(train_df, val_df):
    """
    Trains XGBoost & Random Forest Regressors for season final yield estimation (kg/dönüm).
    """
    print("\n=======================================================")
    print("Training Crop Yield Prediction Models (kg/dönüm)")
    print("=======================================================")
    
    feature_cols = [
        'soil_type', 'is_irrigated', 'peak_ndvi', 'avg_ndvi',
        'avg_ndwi', 'min_ndwi', 'min_soil_moisture', 'avg_soil_moisture',
        'total_precip_mm', 'avg_temp_c'
    ]
    target_col = 'yield_kg_per_da'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    # 1. XGBoost Regressor
    print("\n[1/2] Fitting XGBoost Yield Regressor...")
    xgb_yield = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    xgb_yield.fit(X_train, y_train)
    
    xgb_preds = xgb_yield.predict(X_val)
    xgb_r2 = r2_score(y_val, xgb_preds)
    xgb_mae = mean_absolute_error(y_val, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
    
    print(f"XGBoost Yield Metrics -> R² Score: {xgb_r2:.4f} | MAE: {xgb_mae:.2f} kg/da | RMSE: {xgb_rmse:.2f} kg/da")
    
    # 2. Random Forest Regressor
    print("\n[2/2] Fitting Random Forest Yield Regressor...")
    rf_yield = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    rf_yield.fit(X_train, y_train)
    
    rf_preds = rf_yield.predict(X_val)
    rf_r2 = r2_score(y_val, rf_preds)
    rf_mae = mean_absolute_error(y_val, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
    
    print(f"Random Forest Yield Metrics -> R² Score: {rf_r2:.4f} | MAE: {rf_mae:.2f} kg/da | RMSE: {rf_rmse:.2f} kg/da")
    
    # Feature Importance analysis
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'XGB_Importance': xgb_yield.feature_importances_,
        'RF_Importance': rf_yield.feature_importances_
    }).sort_values('XGB_Importance', ascending=False)
    
    print("\nTop Features for Yield Prediction:")
    print(feature_importance.to_string(index=False))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_yield, 'models/yield_xgboost.joblib')
    joblib.dump(rf_yield, 'models/yield_rf.joblib')
    feature_importance.to_csv('models/yield_feature_importance.csv', index=False)
    
    print("\nModels successfully saved to 'models/' directory.")
    return xgb_yield, rf_yield, {'r2': xgb_r2, 'mae': xgb_mae, 'rmse': xgb_rmse}
