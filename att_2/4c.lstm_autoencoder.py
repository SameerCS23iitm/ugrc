# now we try to reconstruct sequences instead of windows
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve

TRAIN_DIR = "../train/"
OBS_DIR = "obs/rolling_autoencoder/"

# Load data
cscada = pd.read_csv(TRAIN_DIR + "labeled_1s_cscada.csv")

cols = cscada.columns.tolist()
FEATURE_COLS = [col for col in cols if col not in ["label", "time_window"]]

# Split benign and attack
benign_df = cscada[cscada["label"] == 0].copy()
attack_df = cscada[cscada["label"] == 1].copy()

print(f"Benign samples: {len(benign_df)}")
print(f"Attack samples: {len(attack_df)}")

# ================== CREATE SEQUENCES ==================

def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

SEQ_LEN = 10

# Time-aware splitting to reduce leakage risk.
if "time_window" in benign_df.columns:
    benign_df = benign_df.sort_values("time_window").reset_index(drop=True)
if "time_window" in attack_df.columns:
    attack_df = attack_df.sort_values("time_window").reset_index(drop=True)

# Balanced test set (equal benign/attack rows), with benign remainder for train+calibration.
n_test_each = min(len(attack_df), len(benign_df) - (2 * SEQ_LEN + 20))
if n_test_each <= SEQ_LEN:
    raise ValueError("Insufficient data to create balanced test split with sequence length constraints.")

benign_test_df = benign_df.iloc[-n_test_each:].copy()
attack_test_df = attack_df.iloc[-n_test_each:].copy()
benign_remaining_df = benign_df.iloc[:-n_test_each].copy()

calib_frac = 0.15
calib_start_idx = int(len(benign_remaining_df) * (1 - calib_frac))
benign_train_df = benign_remaining_df.iloc[:calib_start_idx].copy()
benign_calib_df = benign_remaining_df.iloc[calib_start_idx:].copy()

if len(benign_train_df) <= SEQ_LEN or len(benign_calib_df) <= SEQ_LEN:
    raise ValueError("Benign train/calibration split is too small for sequence generation.")

print("\nSplit summary (rows):")
print(f"  Benign train: {len(benign_train_df)}")
print(f"  Benign calib: {len(benign_calib_df)}")
print(f"  Benign test : {len(benign_test_df)}")
print(f"  Attack test : {len(attack_test_df)}")

# Normalize using train benign stats only.
scaler = MinMaxScaler()
scaler.fit(benign_train_df[FEATURE_COLS])

X_benign_train = scaler.transform(benign_train_df[FEATURE_COLS])
X_benign_calib = scaler.transform(benign_calib_df[FEATURE_COLS])
X_benign_test = scaler.transform(benign_test_df[FEATURE_COLS])
X_attack_test = scaler.transform(attack_test_df[FEATURE_COLS])

X_benign_train_seq = create_sequences(X_benign_train, SEQ_LEN)
X_benign_calib_seq = create_sequences(X_benign_calib, SEQ_LEN)
X_benign_test_seq = create_sequences(X_benign_test, SEQ_LEN)
X_attack_test_seq = create_sequences(X_attack_test, SEQ_LEN)

# Keep test set balanced at sequence level too.
n_test_seq = min(len(X_benign_test_seq), len(X_attack_test_seq))
if n_test_seq == 0:
    raise ValueError("No test sequences available after split; reduce SEQ_LEN or adjust split.")
X_benign_test_seq = X_benign_test_seq[:n_test_seq]
X_attack_test_seq = X_attack_test_seq[:n_test_seq]

print("\nSequence shapes:")
print("  Benign train seq:", X_benign_train_seq.shape)
print("  Benign calib seq:", X_benign_calib_seq.shape)
print("  Benign test seq :", X_benign_test_seq.shape)
print("  Attack test seq :", X_attack_test_seq.shape)


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

X_benign_tensor = torch.FloatTensor(X_benign_train_seq).to(device)
X_benign_val_tensor = torch.FloatTensor(X_benign_calib_seq).to(device)

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
train_losses = []
val_losses = []
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

    model.eval()
    with torch.no_grad():
        val_recon = model(X_benign_val_tensor)
        val_loss = criterion(val_recon, X_benign_val_tensor).item()
    model.train()

    train_losses.append(avg_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

print("✓ Training complete\n")


# ================== RECONSTRUCTION ==================

model.eval()

X_benign_calib_tensor = torch.FloatTensor(X_benign_calib_seq).to(device)
X_benign_test_tensor = torch.FloatTensor(X_benign_test_seq).to(device)
X_attack_test_tensor = torch.FloatTensor(X_attack_test_seq).to(device)

with torch.no_grad():
    benign_calib_recon = model(X_benign_calib_tensor).cpu().numpy()
    benign_test_recon = model(X_benign_test_tensor).cpu().numpy()
    attack_test_recon = model(X_attack_test_tensor).cpu().numpy()


# ================== RECONSTRUCTION ERROR ==================

benign_calib_mse = np.mean((X_benign_calib_seq - benign_calib_recon) ** 2, axis=(1,2))
benign_test_mse = np.mean((X_benign_test_seq - benign_test_recon) ** 2, axis=(1,2))
attack_test_mse = np.mean((X_attack_test_seq - attack_test_recon) ** 2, axis=(1,2))

# ================== EVALUATION ==================

y_true = np.concatenate([
    np.zeros(len(benign_test_mse)),
    np.ones(len(attack_test_mse))
])

mse_all = np.concatenate([benign_test_mse, attack_test_mse])


# ================== THRESHOLD ==================

threshold = np.percentile(benign_calib_mse, 95)
y_pred = (mse_all > threshold).astype(int)

print("\nThreshold source: benign calibration set")
print("Threshold percentile: 95")
print("Threshold:", threshold)

print(classification_report(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, mse_all))

# ================== PLOTS ==================

plt.figure(figsize=(8, 5))
plt.hist(benign_test_mse, bins=80, alpha=0.5, label="Benign Test")
plt.hist(attack_test_mse, bins=80, alpha=0.5, label="Attack Test")
plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label='Threshold (95% calib benign)')
plt.title("Reconstruction Error Distribution - LSTM Autoencoder")
plt.xlabel("Reconstruction MSE")
plt.ylabel("Count")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OBS_DIR + 'rec_err.png', dpi=150, bbox_inches='tight')
plt.show()

precision, recall, _ = precision_recall_curve(y_true, mse_all)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, linewidth=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve - LSTM Autoencoder')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Benign', 'Attack'],
            yticklabels=['Benign', 'Attack'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - LSTM Autoencoder')
plt.tight_layout()
plt.savefig(OBS_DIR + 'con_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

auc = roc_auc_score(y_true, mse_all)
fpr, tpr, _ = roc_curve(y_true, mse_all)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'LSTM Autoencoder (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - LSTM Autoencoder')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'roc.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, label='Train Loss')
ax.plot(val_losses, label='Val Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training Loss - LSTM Autoencoder')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'losses.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved reconstruction histogram to {OBS_DIR}recon_hist_lstm_autoenc.png")
print(f"Saved confusion matrix to {OBS_DIR}con_matrix_lstm_autoenc.png")
print(f"Saved PR curve to {OBS_DIR}pr_curve_lstm_autoenc.png")
print(f"Saved ROC curve to {OBS_DIR}roc_lstm_autoenc.png")
print(f"Saved training loss curve to {OBS_DIR}losses_lstm_autoenc.png")