import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from xgboost import XGBClassifier
import joblib

# Set PyTorch Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeSeriesDataset(Dataset):
    """
    Constructs sequence windows (e.g., past 4 satellite observations = 20 days)
    to predict future drought stress 14 days ahead.
    """
    def __init__(self, df, sequence_length=4, feature_cols=None):
        if feature_cols is None:
            feature_cols = ['ndvi', 'ndwi', 'temp_c', 'precip_mm', 'soil_moisture', 'soil_type', 'is_irrigated']
            
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.sequences = []
        self.labels = []
        
        # Group by field_id to create valid temporal sequences
        grouped = df.groupby('field_id')
        for _, group in grouped:
            group_sorted = group.sort_values('step')
            features = group_sorted[self.feature_cols].values
            targets = group_sorted['stress_target_14d'].values
            
            for i in range(len(group_sorted) - sequence_length + 1):
                seq = features[i : i + sequence_length]
                label = targets[i + sequence_length - 1]
                self.sequences.append(seq)
                self.labels.append(label)
                
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class DroughtLSTM(nn.Module):
    """
    Deep LSTM Network for 14-Day Drought & Water Stress Early Warning
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super(DroughtLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take the output of the last sequence step
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def train_drought_lstm(train_df, val_df, epochs=30, batch_size=64, lr=0.001):
    print("\n==========================================")
    print("Training PyTorch LSTM Drought Stress Model")
    print("==========================================")
    
    train_dataset = TimeSeriesDataset(train_df)
    val_dataset = TimeSeriesDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = len(train_dataset.feature_cols)
    model = DroughtLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_f1 = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seqs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * seqs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/drought_lstm_best.pth')
            
    print(f"LSTM Training Complete! Best Validation Macro F1: {best_val_f1:.4f}")
    
    # Final evaluation on validation set
    model.load_state_dict(torch.load('models/drought_lstm_best.pth'))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs = seqs.to(device)
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\nClassification Report (LSTM):")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Orta Stres', 'Şiddetli Kuraklık']))
    
    return model, val_f1


def prepare_xgboost_features(df):
    """
    Extracts tabular rolling lag & change features for XGBoost baseline
    """
    df = df.copy().sort_values(['field_id', 'step'])
    
    # Compute 1-step & 2-step differences (deltas)
    df['delta_ndvi_1'] = df.groupby('field_id')['ndvi'].diff(1).fillna(0)
    df['delta_ndwi_1'] = df.groupby('field_id')['ndwi'].diff(1).fillna(0)
    df['delta_soil_1'] = df.groupby('field_id')['soil_moisture'].diff(1).fillna(0)
    
    df['rolling_mean_soil'] = df.groupby('field_id')['soil_moisture'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['rolling_mean_ndwi'] = df.groupby('field_id')['ndwi'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    feature_cols = [
        'ndvi', 'ndwi', 'temp_c', 'precip_mm', 'soil_moisture', 'soil_type', 'is_irrigated',
        'delta_ndvi_1', 'delta_ndwi_1', 'delta_soil_1', 'rolling_mean_soil', 'rolling_mean_ndwi'
    ]
    return df[feature_cols], df['stress_target_14d'], feature_cols


def train_drought_xgboost(train_df, val_df):
    print("\n=============================================")
    print("Training XGBoost Time-Series Drought Classifier")
    print("=============================================")
    
    X_train, y_train, feature_cols = prepare_xgboost_features(train_df)
    X_val, y_val, _ = prepare_xgboost_features(val_df)
    
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    val_preds = xgb_model.predict(X_val)
    
    macro_f1 = f1_score(y_val, val_preds, average='macro')
    print(f"XGBoost Training Complete! Validation Macro F1: {macro_f1:.4f}")
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_val, val_preds, target_names=['Normal', 'Orta Stres', 'Şiddetli Kuraklık']))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/drought_xgboost.joblib')
    
    return xgb_model, macro_f1
