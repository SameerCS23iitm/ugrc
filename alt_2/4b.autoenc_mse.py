import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve


TRAIN_DIR = "../train/"
OBS_DIR = "obs/autoencoder/"
os.makedirs(OBS_DIR, exist_ok=True)


# Load data
cscada = pd.read_csv(TRAIN_DIR + "labeled_1s_cscada.csv")

cols = cscada.columns.tolist()
FEATURE_COLS = [col for col in cols if col not in ["label", "time_window"]]

# Split benign and attack
benign_df = cscada[cscada["label"] == 0].copy()
attack_df = cscada[cscada["label"] == 1].copy()

print(f"Benign samples: {len(benign_df)}")
print(f"Attack samples: {len(attack_df)}")

# Time-aware splitting to reduce leakage risk.
if "time_window" in benign_df.columns:
    benign_df = benign_df.sort_values("time_window").reset_index(drop=True)
if "time_window" in attack_df.columns:
    attack_df = attack_df.sort_values("time_window").reset_index(drop=True)

# Balanced test set: equal benign and attack samples.
n_test_each = min(len(attack_df), max(1, len(benign_df) - 2))
if n_test_each < 1:
    raise ValueError("Insufficient data for balanced test split.")

benign_test_df = benign_df.iloc[-n_test_each:].copy()
attack_test_df = attack_df.iloc[-n_test_each:].copy()
benign_remaining_df = benign_df.iloc[:-n_test_each].copy()

if len(benign_remaining_df) < 10:
    raise ValueError("Too few benign samples left for training/calibration after test split.")

# Calibration split from remaining benign (used only for threshold selection).
calib_frac = 0.15
calib_start_idx = int(len(benign_remaining_df) * (1 - calib_frac))
benign_train_df = benign_remaining_df.iloc[:calib_start_idx].copy()
benign_calib_df = benign_remaining_df.iloc[calib_start_idx:].copy()

print("\nSplit summary:")
print(f"  Benign train: {len(benign_train_df)}")
print(f"  Benign calib: {len(benign_calib_df)}")
print(f"  Benign test : {len(benign_test_df)}")
print(f"  Attack test : {len(attack_test_df)}")

# Normalize features using benign data stats only (prevent data leakage)
scaler = MinMaxScaler()
scaler.fit(benign_train_df[FEATURE_COLS])

X_benign_train = scaler.transform(benign_train_df[FEATURE_COLS])
X_benign_calib = scaler.transform(benign_calib_df[FEATURE_COLS])
X_benign_test = scaler.transform(benign_test_df[FEATURE_COLS])
X_attack_test = scaler.transform(attack_test_df[FEATURE_COLS])

print(f"Normalized X_benign_train shape: {X_benign_train.shape}")
print(f"Normalized X_benign_calib shape: {X_benign_calib.shape}")
print(f"Normalized X_benign_test shape:  {X_benign_test.shape}")
print(f"Normalized X_attack_test shape:  {X_attack_test.shape}")


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
X_benign_train_tensor = torch.FloatTensor(X_benign_train).to(device)
X_benign_val_tensor = torch.FloatTensor(X_benign_calib).to(device)

# Create dataloader
batch_size = 32
dataset = TensorDataset(X_benign_train_tensor)
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
    
    avg_train_loss = total_loss / len(dataloader)

    model.eval()
    with torch.no_grad():
        val_recon = model(X_benign_val_tensor)
        val_loss = criterion(val_recon, X_benign_val_tensor).item()
    model.train()

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

print("✓ Training complete\n")


# ================== COMPUTE RECONSTRUCTION ERROR ==================

model.eval()

# Convert to tensors
X_benign_calib_tensor = torch.FloatTensor(X_benign_calib).to(device)
X_benign_test_tensor = torch.FloatTensor(X_benign_test).to(device)
X_attack_test_tensor = torch.FloatTensor(X_attack_test).to(device)

with torch.no_grad():
    benign_calib_recon = model(X_benign_calib_tensor).cpu().numpy()
    benign_test_recon = model(X_benign_test_tensor).cpu().numpy()
    attack_test_recon = model(X_attack_test_tensor).cpu().numpy()

# Sample-level reconstruction error
benign_calib_mse = np.mean((X_benign_calib - benign_calib_recon) ** 2, axis=1)
benign_test_mse = np.mean((X_benign_test - benign_test_recon) ** 2, axis=1)
attack_test_mse = np.mean((X_attack_test - attack_test_recon) ** 2, axis=1)

plt.hist(benign_test_mse, bins=100, alpha=0.5, label="Benign Test")
plt.hist(attack_test_mse, bins=100, alpha=0.5, label="Attack Test")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()

# Combine
y_true = np.concatenate([
    np.zeros(len(benign_test_mse)),
    np.ones(len(attack_test_mse))
])

mse_all = np.concatenate([benign_test_mse, attack_test_mse])

# ================== THRESHOLD + METRICS/PLOTS ==================

# Threshold is selected ONLY on benign calibration errors.
best_threshold = np.percentile(benign_calib_mse, 95)
y_pred = (mse_all > best_threshold).astype(int)

print(f"Threshold from benign calibration (95th percentile): {best_threshold:.6f}")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack']))

auc = roc_auc_score(y_true, mse_all)
print(f"ROC-AUC Score: {auc:.4f}")

precision, recall, _ = precision_recall_curve(y_true, mse_all)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, linewidth=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve - Autoencoder (MSE)')
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
ax.set_title('Confusion Matrix - Autoencoder (MSE)')
plt.tight_layout()
plt.savefig(OBS_DIR + 'con_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

fpr, tpr, _ = roc_curve(y_true, mse_all)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'Autoencoder (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Autoencoder (MSE)')
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
ax.set_title('Training Loss - Autoencoder (MSE)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'losses.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved confusion matrix to {OBS_DIR}con_matrix.png")
print(f"Saved PR curve to {OBS_DIR}pr_curve.png")
print(f"Saved ROC curve to {OBS_DIR}roc.png")
print(f"Saved training loss curve to {OBS_DIR}losses.png")




