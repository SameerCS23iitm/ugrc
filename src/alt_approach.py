import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

FEATURE_COLS = [
    "packet_count", "total_bytes", "mean_packet_size", "std_packet_size",
    "iat_mean", "iat_std", "min_iat", "max_iat",
    "unique_func_codes", "read_count", "write_count", "exception_count"
]
SEQ_LEN = 20
DATA_DIR = "../train/"

# ── Step 1: Fit scaler on benign data only ────────────────────────────────────
benign = pd.read_csv(DATA_DIR + "1s_benign_flows.csv")
# Filter out multicast/broadcast addresses
VALID_IPS = [
    "185.175.0.2", "185.175.0.3", "185.175.0.4",
    "185.175.0.5", "185.175.0.6", "185.175.0.7", "185.175.0.8"
]

benign = benign[
    benign["ip.src"].isin(VALID_IPS) & 
    benign["ip.dst"].isin(VALID_IPS)
]
# print(f"Rows after filtering: {len(benign)}")
scaler = MinMaxScaler()
scaler.fit(benign[FEATURE_COLS])
joblib.dump(scaler, DATA_DIR + "scaler_benign_1s.pkl")

# ── Step 2: Scale and build sequences from benign ─────────────────────────────
benign[FEATURE_COLS] = scaler.transform(benign[FEATURE_COLS])

def build_sequences(df, seq_len=SEQ_LEN):
    X = []
    for (src, dst), group in df.groupby(["ip.src", "ip.dst"]):
        group = group.sort_values("time_window").reset_index(drop=True)
        features = group[FEATURE_COLS].values
        if len(group) < seq_len:
            continue
        for i in range(len(group) - seq_len):
            X.append(features[i:i+seq_len])
    return np.array(X)

X_benign = build_sequences(benign)
print(f"Benign sequences: {X_benign.shape}")

# 80/20 split for train/val
split = int(len(X_benign) * 0.8)
X_train = X_benign[:split]
X_val   = X_benign[split:]
print(f"Train: {X_train.shape} | Val: {X_val.shape}")

np.save(DATA_DIR + "X_train_benign_1s.npy", X_train)
np.save(DATA_DIR + "X_val_benign_1s.npy",   X_val)

# ── Step 3: Scale cscada using benign scaler (important!) ────────────────────
cscada = pd.read_csv(DATA_DIR + "labeled_1s_cscada.csv")
cscada = cscada[
    cscada["ip.src"].isin(VALID_IPS) & 
    cscada["ip.dst"].isin(VALID_IPS)
]
cscada[FEATURE_COLS] = scaler.transform(cscada[FEATURE_COLS])

X_cscada = build_sequences(cscada)
y_cscada = []
for (src, dst), group in cscada.groupby(["ip.src", "ip.dst"]):
    group = group.sort_values("time_window").reset_index(drop=True)
    if len(group) < SEQ_LEN:
        continue
    for i in range(len(group) - SEQ_LEN):
        y_cscada.append(group["label"].iloc[i+SEQ_LEN])
y_cscada = np.array(y_cscada)

print(f"Cscada sequences: {X_cscada.shape}")
print(f"Attack sequences: {y_cscada.sum()} ({100*y_cscada.sum()/len(y_cscada):.1f}%)")

np.save(DATA_DIR + "X_cscada_1s.npy", X_cscada)
np.save(DATA_DIR + "y_cscada_1s.npy", y_cscada)