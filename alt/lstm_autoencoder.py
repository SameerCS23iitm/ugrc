# now we try to reconstruct sequences instead of windows
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

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

# ================== CREATE SEQUENCES ==================

def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

SEQ_LEN = 10

X_benign_seq = create_sequences(X_benign, SEQ_LEN)
X_attack_seq = create_sequences(X_attack, SEQ_LEN)

print("Benign sequence shape:", X_benign_seq.shape)
print("Attack sequence shape:", X_attack_seq.shape)


# ================== LSTM AUTOENCODER ==================

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=32):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            batch_first=True
        )

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


# ================== DEVICE ==================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ================== DATA ==================

X_benign_tensor = torch.FloatTensor(X_benign_seq).to(device)

batch_size = 32
dataset = TensorDataset(X_benign_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ================== MODEL ==================

input_dim = len(FEATURE_COLS)
model = LSTMAutoencoder(n_features=input_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ================== TRAIN ==================

epochs = 100
print(f"\nTraining LSTM autoencoder on benign data for {epochs} epochs...")

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
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

print("✓ Training complete\n")


# ================== RECONSTRUCTION ==================

model.eval()

X_benign_tensor = torch.FloatTensor(X_benign_seq).to(device)
X_attack_tensor = torch.FloatTensor(X_attack_seq).to(device)

with torch.no_grad():
    benign_recon = model(X_benign_tensor).cpu().numpy()
    attack_recon = model(X_attack_tensor).cpu().numpy()


# ================== RECONSTRUCTION ERROR ==================

benign_mse = np.mean((X_benign_seq - benign_recon) ** 2, axis=(1,2))
attack_mse = np.mean((X_attack_seq - attack_recon) ** 2, axis=(1,2))

# ================== EVALUATION ==================

y_true = np.concatenate([
    np.zeros(len(benign_mse)),
    np.ones(len(attack_mse))
])

mse_all = np.concatenate([benign_mse, attack_mse])


# ================== THRESHOLD ==================

for p in [95, 90, 85, 80]:
    threshold = np.percentile(benign_mse, p)

    y_pred = (mse_all > threshold).astype(int)

    print(f"\nThreshold percentile: {p}")
    print("Threshold:", threshold)

    print(classification_report(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, mse_all))