import os
import json
import optuna
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, r2_score

from src.models.drought_model import prepare_xgboost_features

# Suppress verbose optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_drought_xgboost(train_df, val_df, n_trials=15):
    """
    Automated Hyperparameter Optimization for XGBoost Drought Classifier via Optuna.
    """
    print(f"\nRunning Optuna Hyperparameter Optimization ({n_trials} trials)...")
    
    X_train, y_train, _ = prepare_xgboost_features(train_df)
    X_val, y_val, _ = prepare_xgboost_features(val_df)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 250),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average='macro')
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Optuna Optimization Complete!")
    print(f"Best Trial Score (Macro F1): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    os.makedirs('models', exist_ok=True)
    with open('models/best_hyperparameters.json', 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2)
        
    return study.best_params
