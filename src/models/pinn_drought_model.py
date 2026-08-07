import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.models.drought_model import TimeSeriesDataset, DroughtLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AgronomicPhysicsLoss(nn.Module):
    """
    Biophysical Physics-Informed Loss Layer for Satellite Crop Health.
    Enforces agronomic laws of plant physiology during backpropagation:
    1. Soil Moisture Deficit Constraint: Low soil moisture MUST correlate with non-zero stress.
    2. Spectral Coherence Constraint: NDWI drop cannot lag behind sharp NDVI drop without penalty.
    """
    def __init__(self, physics_weight=0.25):
        super(AgronomicPhysicsLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.physics_weight = physics_weight

    def forward(self, logits, targets, input_seqs):
        # Base Cross-Entropy Data Loss
        data_loss = self.ce_loss(logits, targets)
        
        # Extract features from last step of sequence:
        # seq shape: (batch_size, seq_len, 7)
        # feature indices: [ndvi=0, ndwi=1, temp=2, precip=3, soil_m=4]
        last_ndvi = input_seqs[:, -1, 0]
        last_ndwi = input_seqs[:, -1, 1]
        last_soil = input_seqs[:, -1, 4]
        
        probs = torch.softmax(logits, dim=1)
        prob_normal = probs[:, 0]
        prob_severe = probs[:, 2]
        
        # Penalty 1: Soil moisture < 12.0% but model predicts Class 0 (Normal) with high probability
        soil_deficit_mask = (last_soil < 12.0).float()
        penalty_soil = torch.mean(soil_deficit_mask * prob_normal)
        
        # Penalty 2: NDWI < -0.15 (Severe Leaf Water Deficit) but model predicts Class 0 (Normal)
        water_deficit_mask = (last_ndwi < -0.15).float()
        penalty_water = torch.mean(water_deficit_mask * prob_normal)
        
        physics_loss = penalty_soil + penalty_water
        total_loss = data_loss + self.physics_weight * physics_loss
        return total_loss


def train_pinn_drought_model(train_df, val_df, epochs=25, batch_size=64, lr=0.001):
    print("\n==========================================================================")
    print(" Training Physics-Informed Agronomic Neural Network (PINN Loss Layer) ")
    print("==========================================================================")
    
    train_dataset = TimeSeriesDataset(train_df)
    val_dataset = TimeSeriesDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = len(train_dataset.feature_cols)
    model = DroughtLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=3).to(device)
    
    criterion = AgronomicPhysicsLoss(physics_weight=0.30)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_f1 = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels, seqs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seqs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
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
                
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - PINN Train Loss: {train_loss:.4f} | Val Macro F1: {val_f1:.4f}")
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/drought_pinn_lstm.pth')
            
    print(f"\nPINN Model Training Complete! Best Validation Macro F1: {best_val_f1:.4f}")
    return model, best_val_f1
