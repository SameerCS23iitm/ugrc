# now we try to reconstruct sequences instead of windows
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve

TRAIN_DIR = "../train/"
OBS_DIR   = "obs/rolling_autoencoder/"
os.makedirs(OBS_DIR, exist_ok=True)

SEQ_LEN    = 10
BATCH_SIZE = 32

# ─────────────────────────────────────────────
# ① LOAD & SPLIT ROWS (no sequences yet)
# ─────────────────────────────────────────────
cscada = pd.read_csv(TRAIN_DIR + "labeled_1s_cscada.csv")

cols         = cscada.columns.tolist()
FEATURE_COLS = [col for col in cols if col not in ["label", "time_window"]]

benign_df = cscada[cscada["label"] == 0].copy()
attack_df = cscada[cscada["label"] == 1].copy()

del cscada   # free the full DataFrame immediately
gc.collect()

print(f"Benign samples: {len(benign_df)}")
print(f"Attack samples: {len(attack_df)}")

# Time-aware sort
if "time_window" in benign_df.columns:
    benign_df = benign_df.sort_values("time_window").reset_index(drop=True)
if "time_window" in attack_df.columns:
    attack_df = attack_df.sort_values("time_window").reset_index(drop=True)

# Balanced test set
n_test_each = min(len(attack_df), len(benign_df) - (2 * SEQ_LEN + 20))
if n_test_each <= SEQ_LEN:
    raise ValueError("Insufficient data for balanced test split with sequence length constraints.")

benign_test_df      = benign_df.iloc[-n_test_each:].copy()
attack_test_df      = attack_df.iloc[-n_test_each:].copy()
benign_remaining_df = benign_df.iloc[:-n_test_each].copy()

del benign_df, attack_df
gc.collect()

calib_frac      = 0.15
calib_start_idx = int(len(benign_remaining_df) * (1 - calib_frac))
benign_train_df = benign_remaining_df.iloc[:calib_start_idx].copy()
benign_calib_df = benign_remaining_df.iloc[calib_start_idx:].copy()

del benign_remaining_df
gc.collect()

if len(benign_train_df) <= SEQ_LEN or len(benign_calib_df) <= SEQ_LEN:
    raise ValueError("Benign train/calibration split is too small for sequence generation.")

print("\nSplit summary (rows):")
print(f"  Benign train: {len(benign_train_df)}")
print(f"  Benign calib: {len(benign_calib_df)}")
print(f"  Benign test : {len(benign_test_df)}")
print(f"  Attack test : {len(attack_test_df)}")

# ─────────────────────────────────────────────
# ② SCALE  (fit on benign train only)
# ─────────────────────────────────────────────
scaler = MinMaxScaler()
scaler.fit(benign_train_df[FEATURE_COLS])

X_benign_train = scaler.transform(benign_train_df[FEATURE_COLS]).astype(np.float32)
X_benign_calib = scaler.transform(benign_calib_df[FEATURE_COLS]).astype(np.float32)
X_benign_test  = scaler.transform(benign_test_df[FEATURE_COLS]).astype(np.float32)
X_attack_test  = scaler.transform(attack_test_df[FEATURE_COLS]).astype(np.float32)

del benign_train_df, benign_calib_df, benign_test_df, attack_test_df
gc.collect()

# ─────────────────────────────────────────────
# ③ SEQUENCE GENERATOR  (training only)
#
#    Where are the train sequences?
#    ──────────────────────────────
#    They are never fully built. __getitem__
#    slices one window at a time from
#    X_benign_train (a 2-D float32 array).
#    Peak RAM = one batch of sequences.
#
#    Calib / test sequences ARE materialised
#    below — those slices are small (balanced,
#    capped at n_test_each rows) so it's fine.
# ─────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, features, seq_len=SEQ_LEN):
        # features: (n_rows, n_features) float32 numpy array
        self.features = torch.from_numpy(features)  # stays on CPU
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        seq = self.features[idx : idx + self.seq_len]
        return seq, seq   # autoencoder: target == input

train_dataset = SequenceDataset(X_benign_train, SEQ_LEN)
dataloader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"\nTrain sequences (virtual): {len(train_dataset)}")

# ─────────────────────────────────────────────
# ④ BUILD CALIB / TEST SEQUENCES
#    These are small — materialise normally.
# ─────────────────────────────────────────────
def create_sequences(data, seq_len=SEQ_LEN):
    return np.stack([data[i:i+seq_len] for i in range(len(data) - seq_len)])

X_benign_calib_seq = create_sequences(X_benign_calib, SEQ_LEN)
X_benign_test_seq  = create_sequences(X_benign_test,  SEQ_LEN)
X_attack_test_seq  = create_sequences(X_attack_test,  SEQ_LEN)

# Keep test set balanced at sequence level
n_test_seq = min(len(X_benign_test_seq), len(X_attack_test_seq))
if n_test_seq == 0:
    raise ValueError("No test sequences available; reduce SEQ_LEN or adjust split.")
X_benign_test_seq = X_benign_test_seq[:n_test_seq]
X_attack_test_seq = X_attack_test_seq[:n_test_seq]

print("\nSequence shapes:")
print(f"  Benign train seq (generator): ({len(train_dataset)}, {SEQ_LEN}, {len(FEATURE_COLS)})")
print(f"  Benign calib seq: {X_benign_calib_seq.shape}")
print(f"  Benign test seq : {X_benign_test_seq.shape}")
print(f"  Attack test seq : {X_attack_test_seq.shape}")

# ─────────────────────────────────────────────
# ⑤ MODEL
# ─────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=32):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=n_features,  hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim,  hidden_size=n_features, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(FEATURE_COLS)

model     = LSTMAutoencoder(n_features=input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nUsing device: {device}")

# ─────────────────────────────────────────────
# ⑥ CALIB TENSOR  (for val loss each epoch)
# ─────────────────────────────────────────────
X_benign_val_tensor = torch.FloatTensor(X_benign_calib_seq).to(device)

# ─────────────────────────────────────────────
# ⑦ TRAIN
# ─────────────────────────────────────────────
epochs = 100
print(f"\nTraining LSTM autoencoder on benign data for {epochs} epochs...")

train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        reconstructed = model(batch_x)
        loss = criterion(reconstructed, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    model.eval()
    with torch.no_grad():
        val_recon = model(X_benign_val_tensor)
        val_loss  = criterion(val_recon, X_benign_val_tensor).item()

    train_losses.append(avg_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

print("✓ Training complete\n")

# ─────────────────────────────────────────────
# ⑧ RECONSTRUCTION ERRORS
# ─────────────────────────────────────────────
model.eval()

X_benign_calib_tensor = torch.FloatTensor(X_benign_calib_seq).to(device)
X_benign_test_tensor  = torch.FloatTensor(X_benign_test_seq).to(device)
X_attack_test_tensor  = torch.FloatTensor(X_attack_test_seq).to(device)

with torch.no_grad():
    benign_calib_recon = model(X_benign_calib_tensor).cpu().numpy()
    benign_test_recon  = model(X_benign_test_tensor).cpu().numpy()
    attack_test_recon  = model(X_attack_test_tensor).cpu().numpy()

benign_calib_mse = np.mean((X_benign_calib_seq - benign_calib_recon) ** 2, axis=(1, 2))
benign_test_mse  = np.mean((X_benign_test_seq  - benign_test_recon)  ** 2, axis=(1, 2))
attack_test_mse  = np.mean((X_attack_test_seq  - attack_test_recon)  ** 2, axis=(1, 2))

# ─────────────────────────────────────────────
# ⑨ THRESHOLD + METRICS
# ─────────────────────────────────────────────
y_true  = np.concatenate([np.zeros(len(benign_test_mse)), np.ones(len(attack_test_mse))])
mse_all = np.concatenate([benign_test_mse, attack_test_mse])

threshold = np.percentile(benign_calib_mse, 95)
y_pred    = (mse_all > threshold).astype(int)

print("\nThreshold source: benign calibration set")
print(f"Threshold percentile: 95")
print(f"Threshold: {threshold:.6f}")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack']))
print(f"ROC-AUC: {roc_auc_score(y_true, mse_all):.4f}")

# ─────────────────────────────────────────────
# ⑩ PLOTS
# ─────────────────────────────────────────────
auc = roc_auc_score(y_true, mse_all)

# Reconstruction error histogram
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

# Precision-Recall
precision, recall, _ = precision_recall_curve(y_true, mse_all)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, linewidth=2)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve - LSTM Autoencoder')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Benign', 'Attack'],
            yticklabels=['Benign', 'Attack'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - LSTM Autoencoder')
plt.tight_layout()
plt.savefig(OBS_DIR + 'con_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_true, mse_all)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'LSTM Autoencoder (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - LSTM Autoencoder')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'roc.png', dpi=150, bbox_inches='tight')
plt.show()

# Training loss
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, label='Train Loss')
ax.plot(val_losses,   label='Val Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('Training Loss - LSTM Autoencoder')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'losses.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved all plots to {OBS_DIR}")