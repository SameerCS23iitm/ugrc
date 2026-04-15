#trying to get relative weights of features in the anomaly detection.

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

TRAIN_DIR = "../train/"

cscada = pd.read_csv(TRAIN_DIR + "labeled_1s_cscada.csv")

FEATURE_COLS = [
    "packet_count", "total_bytes", "mean_packet_size", "std_packet_size",
    "iat_mean", "iat_std", "min_iat", "max_iat",
    "unique_func_codes", "read_count", "write_count", "exception_count"
]

# X = features, y = labels
X = cscada[FEATURE_COLS]
y = cscada["label"]   # attack / benign labels

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance (weights)
weights = rf.feature_importances_

# Normalize weights (optional but recommended)
weights = weights / np.sum(weights)

# Map feature -> weight
feature_weights = dict(zip(FEATURE_COLS, weights))

for key, value in feature_weights.items():
    print(f"{key}: {value}")

