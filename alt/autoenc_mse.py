import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

TRAIN_DIR = "../train/"

# Load data
cscada = pd.read_csv(TRAIN_DIR + "labeled_1s_cscada.csv")

FEATURE_COLS = [
    "packet_count", "total_bytes", "mean_packet_size", "std_packet_size",
    "iat_mean", "iat_std", "min_iat", "max_iat",
    "unique_func_codes", "read_count", "write_count", "exception_count"
]

# Split benign and attack
benign_df = cscada[cscada["label"] == 0].copy()
attack_df = cscada[cscada["label"] == 1].copy()

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
    def __init__(self, input_dim):
        super(SimpleAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()
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
model = SimpleAutoencoder(input_dim=input_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 100
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

# Convert to tensors
X_benign_tensor = torch.FloatTensor(X_benign).to(device)
X_attack_tensor = torch.FloatTensor(X_attack).to(device)

with torch.no_grad():
    benign_recon = model(X_benign_tensor).cpu().numpy()
    attack_recon = model(X_attack_tensor).cpu().numpy()

# Sample-level reconstruction error
benign_mse = np.mean((X_benign - benign_recon) ** 2, axis=1)
attack_mse = np.mean((X_attack - attack_recon) ** 2, axis=1)

import matplotlib.pyplot as plt

plt.hist(benign_mse, bins=100, alpha=0.5, label="Benign")
plt.hist(attack_mse, bins=100, alpha=0.5, label="Attack")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()

# Combine
y_true = np.concatenate([
    np.zeros(len(benign_mse)),
    np.ones(len(attack_mse))
])

mse_all = np.concatenate([benign_mse, attack_mse])

# ================== THRESHOLD ==================

from sklearn.metrics import classification_report, roc_auc_score
for p in [90, 85, 80]:
    threshold = np.percentile(benign_mse, p)
    y_pred = (mse_all > threshold).astype(int)

    print(f"\nThreshold percentile: {p}")
    print(classification_report(y_true, y_pred))




