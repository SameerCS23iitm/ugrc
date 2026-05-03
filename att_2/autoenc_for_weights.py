import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

TRAIN_DIR = "../train/"
CSCADA = "labeled_1s_cscada.csv"
EXTERNAL = "labeled_1s_external.csv"

# Load data
data = pd.read_csv(TRAIN_DIR + EXTERNAL)  # or CSCADA, depending on which you want to analyze

cols = data.columns.tolist()
FEATURE_COLS = [col for col in cols if col not in ["label", "time_window"]]


# Split benign and attack
benign_df = data[data["label"] == 0].copy()
attack_df = data[data["label"] == 1].copy()

print(f"Benign samples: {len(benign_df)}")
print(f"Attack samples: {len(attack_df)}")

# Normalize features using benign data stats only (prevent data leakage)
scaler = MinMaxScaler()
scaler.fit(benign_df[FEATURE_COLS])

X_benign = scaler.transform(benign_df[FEATURE_COLS])
X_attack = scaler.transform(attack_df[FEATURE_COLS])

print(f"Normalized X_benign shape: {X_benign.shape}")
print(f"Normalized X_attack shape: {X_attack.shape}")


# ================== AUTOENCODER ==================

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Normalize output to [0,1] since input is MinMax-scaled
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert to tensors
X_benign_tensor = torch.FloatTensor(X_benign).to(device)
# Create dataloader
batch_size = 32
dataset = TensorDataset(X_benign_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
input_dim = len(FEATURE_COLS)
model = SimpleAutoencoder(input_dim=input_dim, hidden_dim=8).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 50
print(f"\nTraining autoencoder on benign data for {epochs} epochs...")

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_data, in dataloader:
        optimizer.zero_grad()
        reconstructed = model(batch_data)
        loss = criterion(reconstructed, batch_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

print("✓ Training complete\n")


# ================== COMPUTE RECONSTRUCTION ERROR ==================

model.eval()

# Reconstruction error per feature for all samples (benign + attack)
X_all = np.vstack([X_benign, X_attack])
X_all_tensor = torch.FloatTensor(X_all).to(device)

with torch.no_grad():
    reconstructed_all = model(X_all_tensor).cpu().numpy()

# Per-feature MSE: (input - output)^2
per_feature_error_all = (X_all - reconstructed_all) ** 2

# Separate by label
benign_size = len(X_benign)
per_feature_error_benign = per_feature_error_all[:benign_size]
per_feature_error_attack = per_feature_error_all[benign_size:]

# Average per-feature error on attack samples → feature importance
feature_weights_array = per_feature_error_attack.mean(axis=0)

# Normalize to sum to 1
feature_weights_array = feature_weights_array / feature_weights_array.sum()

# Dictionary mapping
feature_weights = dict(zip(FEATURE_COLS, feature_weights_array))

print("Feature Importance Weights (from Attack Reconstruction Error):")
print("=" * 60)
for feature, weight in feature_weights.items():
    print(f"  {feature:30s}: {weight:.6f}")
print("=" * 60)
print(f"Sum of weights: {sum(feature_weights.values()):.6f}\n")