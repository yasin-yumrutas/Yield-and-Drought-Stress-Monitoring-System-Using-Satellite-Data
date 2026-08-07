import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.drought_model import TimeSeriesDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """
    Injects positional information into input sequence embeddings.
    """
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class AgriTemporalTransformer(nn.Module):
    """
    Advanced Multi-Head Self-Attention Temporal Transformer for Satellite Crop Health
    and 14-Day Drought Stress Early Warning.
    """
    def __init__(self, input_dim=7, d_model=64, nhead=4, num_layers=2, num_classes=3, dropout=0.2):
        super(AgriTemporalTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out = self.input_projection(x)
        out = self.pos_encoder(out)
        
        # Pass through Transformer Encoder
        out = self.transformer_encoder(out)
        
        # Global Average Pooling across sequence steps
        out = torch.mean(out, dim=1)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        return out


def train_drought_transformer(train_df, val_df, epochs=30, batch_size=64, lr=0.0008):
    print("\n========================================================")
    print("Training Advanced PyTorch Temporal Transformer (Attention)")
    print("========================================================")
    
    train_dataset = TimeSeriesDataset(train_df)
    val_dataset = TimeSeriesDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = len(train_dataset.feature_cols)
    model = AgriTemporalTransformer(input_dim=input_dim, d_model=64, nhead=4, num_layers=2, num_classes=3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
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
        scheduler.step(val_f1)
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/drought_transformer_best.pth')
            
    print(f"\nTemporal Transformer Training Complete! Best Validation Macro F1: {best_val_f1:.4f}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('models/drought_transformer_best.pth'))
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
            
    print("\nClassification Report (Temporal Transformer):")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Orta Stres', 'Şiddetli Kuraklık']))
    
    return model, best_val_f1
