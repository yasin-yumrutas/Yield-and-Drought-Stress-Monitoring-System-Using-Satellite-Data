import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, r2_score, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance in extreme drought detection.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)


class MultiTaskDataset(Dataset):
    """
    Dataset loader for simultaneous Multi-Task Learning:
    Returns (sequence_features, drought_stress_label, seasonal_yield_ground_truth).
    """
    def __init__(self, df_ts, df_yield, sequence_length=4):
        self.sequence_length = sequence_length
        self.feature_cols = ['ndvi', 'ndwi', 'temp_c', 'precip_mm', 'soil_moisture', 'soil_type', 'is_irrigated']
        self.sequences = []
        self.stress_labels = []
        self.yield_targets = []
        
        # Merge seasonal final yield into time-series rows
        yield_map = df_yield.set_index('field_id')['yield_kg_per_da'].to_dict()
        
        grouped = df_ts.groupby('field_id')
        for f_id, group in grouped:
            group_sorted = group.sort_values('step')
            features = group_sorted[self.feature_cols].values
            stress_targets = group_sorted['stress_target_14d'].values
            final_yield = yield_map.get(f_id, 500.0)
            
            for i in range(len(group_sorted) - sequence_length + 1):
                seq = features[i : i + sequence_length]
                label = stress_targets[i + sequence_length - 1]
                self.sequences.append(seq)
                self.stress_labels.append(label)
                self.yield_targets.append(final_yield)
                
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.stress_labels = torch.tensor(np.array(self.stress_labels), dtype=torch.long)
        self.yield_targets = torch.tensor(np.array(self.yield_targets), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.stress_labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.stress_labels[idx], self.yield_targets[idx]


class MultiTaskAgriNet(nn.Module):
    """
    Unified Multi-Task Deep Neural Network architecture:
    Shared Transformer Backbone -> Dual Task Heads (Drought Classification + Yield Regression).
    """
    def __init__(self, input_dim=7, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(MultiTaskAgriNet, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.shared_backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Task Head 1: 14-Day Drought Stress Classifier (3 classes)
        self.drought_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )
        
        # Task Head 2: Season Crop Yield Regressor (Continuous kg/da)
        self.yield_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        feat = self.input_projection(x)
        feat = self.shared_backbone(feat)
        
        # Global Average Pooling
        pooled_feat = torch.mean(feat, dim=1)
        
        drought_logits = self.drought_head(pooled_feat)
        yield_pred = self.yield_head(pooled_feat)
        return drought_logits, yield_pred


def train_multitask_model(df_ts, df_yield_train, df_ts_val, df_yield_val, epochs=25, batch_size=64, lr=0.0008):
    print("\n==========================================================================")
    print(" Training Unified Multi-Task Deep Learning Model (Focal Loss + Shared Transformer) ")
    print("==========================================================================")
    
    train_dataset = MultiTaskDataset(df_ts, df_yield_train)
    val_dataset = MultiTaskDataset(df_ts_val, df_yield_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MultiTaskAgriNet(input_dim=7, d_model=64, nhead=4, num_layers=2).to(device)
    
    focal_criterion = FocalLoss(gamma=2.0)
    mse_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_score = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, stress_targets, yield_targets in train_loader:
            seqs = seqs.to(device)
            stress_targets = stress_targets.to(device)
            yield_targets = yield_targets.to(device)
            
            optimizer.zero_grad()
            drought_logits, yield_preds = model(seqs)
            
            loss_stress = focal_criterion(drought_logits, stress_targets)
            loss_yield = mse_criterion(yield_preds / 500.0, yield_targets / 500.0)  # Normalized MSE
            
            # Joint Multi-Task Loss
            total_loss = loss_stress + 1.2 * loss_yield
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * seqs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        all_stress_preds = []
        all_stress_labels = []
        all_yield_preds = []
        all_yield_labels = []
        
        with torch.no_grad():
            for seqs, stress_targets, yield_targets in val_loader:
                seqs = seqs.to(device)
                drought_logits, yield_preds = model(seqs)
                
                preds = torch.argmax(drought_logits, dim=1)
                all_stress_preds.extend(preds.cpu().numpy())
                all_stress_labels.extend(stress_targets.numpy())
                all_yield_preds.extend(yield_preds.cpu().numpy().flatten())
                all_yield_labels.extend(yield_targets.numpy().flatten())
                
        val_f1 = f1_score(all_stress_labels, all_stress_preds, average='macro')
        val_r2 = r2_score(all_yield_labels, all_yield_preds)
        val_mae = mean_absolute_error(all_yield_labels, all_yield_preds)
        
        combined_score = val_f1 + val_r2
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Multi-Task Loss: {train_loss:.4f} | Stress F1: {val_f1:.4f} | Yield R²: {val_r2:.4f} | MAE: {val_mae:.1f} kg/da")
            
        if combined_score > best_val_score:
            best_val_score = combined_score
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/multitask_agri_net_best.pth')
            
    print(f"\nMulti-Task Neural Network Complete! Best Joint Score: {best_val_score:.4f}")
    return model, val_f1, val_r2
