import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.models.drought_model import DroughtLSTM, prepare_xgboost_features, TimeSeriesDataset
from src.models.advanced_transformer import AgriTemporalTransformer
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DroughtEnsembleMetaLearner:
    """
    Hybrid Ensemble Meta-Learner combining PyTorch Transformer, PyTorch LSTM,
    and XGBoost probability outputs for ultra-robust drought early warning.
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.lstm_model = None
        self.transformer_model = None
        self.xgb_model = None
        self.meta_learner = None
        self.weights = [0.35, 0.35, 0.30]  # Initial weights for Transformer, LSTM, XGBoost
        
    def load_base_models(self):
        # 1. Load LSTM
        lstm_path = os.path.join(self.models_dir, 'drought_lstm_best.pth')
        if os.path.exists(lstm_path):
            self.lstm_model = DroughtLSTM(input_dim=7, hidden_dim=64, num_layers=2, num_classes=3).to(device)
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
            self.lstm_model.eval()
            
        # 2. Load Transformer
        tf_path = os.path.join(self.models_dir, 'drought_transformer_best.pth')
        if os.path.exists(tf_path):
            self.transformer_model = AgriTemporalTransformer(input_dim=7, d_model=64, nhead=4, num_layers=2, num_classes=3).to(device)
            self.transformer_model.load_state_dict(torch.load(tf_path, map_location=device))
            self.transformer_model.eval()
            
        # 3. Load XGBoost
        xgb_path = os.path.join(self.models_dir, 'drought_xgboost.joblib')
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)

    def _get_base_predictions_proba(self, df):
        """
        Extracts probability vectors from all 3 base models for a dataset.
        """
        dataset = TimeSeriesDataset(df)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        # 1. PyTorch LSTM Probas
        lstm_probs = []
        with torch.no_grad():
            for seqs, _ in loader:
                seqs = seqs.to(device)
                logits = self.lstm_model(seqs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                lstm_probs.append(probs)
        lstm_probs = np.vstack(lstm_probs)
        
        # 2. PyTorch Transformer Probas
        tf_probs = []
        with torch.no_grad():
            for seqs, _ in loader:
                seqs = seqs.to(device)
                logits = self.transformer_model(seqs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                tf_probs.append(probs)
        tf_probs = np.vstack(tf_probs)
        
        # 3. XGBoost Probas (Requires Tabular Lag Features)
        X_xgb, _, _ = prepare_xgboost_features(df)
        # Sequence dataset trims initial sequence_length - 1 rows per field
        # Align XGBoost predictions with sequence targets
        seq_len = dataset.sequence_length
        aligned_indices = []
        grouped = df.groupby('field_id')
        for _, group in grouped:
            group_indices = group.index[seq_len - 1:]
            aligned_indices.extend(group_indices)
            
        X_xgb_aligned = X_xgb.loc[aligned_indices]
        xgb_probs = self.xgb_model.predict_proba(X_xgb_aligned)
        
        # Extract targets
        labels = dataset.labels.numpy()
        
        return tf_probs, lstm_probs, xgb_probs, labels

    def fit_ensemble(self, train_df, val_df):
        print("\n========================================================")
        print(" Training Hybrid Stacking Ensemble Meta-Learner ")
        print("========================================================")
        
        self.load_base_models()
        tf_p_tr, lstm_p_tr, xgb_p_tr, y_tr = self._get_base_predictions_proba(train_df)
        tf_p_val, lstm_p_val, xgb_p_val, y_val = self._get_base_predictions_proba(val_df)
        
        # Stack probabilities as meta-features: shape (N, 9)
        meta_X_train = np.hstack([tf_p_tr, lstm_p_tr, xgb_p_tr])
        meta_X_val = np.hstack([tf_p_val, lstm_p_val, xgb_p_val])
        
        # Fit Logistic Regression Meta-Learner
        self.meta_learner = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        self.meta_learner.fit(meta_X_train, y_tr)
        
        val_preds = self.meta_learner.predict(meta_X_val)
        ensemble_f1 = f1_score(y_val, val_preds, average='macro')
        ensemble_acc = accuracy_score(y_val, val_preds)
        
        print(f"Hybrid Ensemble Meta-Learner Fitted!")
        print(f"Validation Accuracy: {ensemble_acc*100:.2f}% | Macro F1 Score: {ensemble_f1:.4f}")
        print("\nClassification Report (Hybrid Ensemble):")
        print(classification_report(y_val, val_preds, target_names=['Normal', 'Orta Stres', 'Şiddetli Kuraklık']))
        
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.meta_learner, os.path.join(self.models_dir, 'drought_ensemble_meta.joblib'))
        return self.meta_learner, ensemble_f1
