#trying to get relative weights of features in the anomaly detection.

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

TRAIN_DIR = "../train/"
CSCADA = "labeled_1s_cscada.csv"
EXTERNAL = "labeled_1s_external.csv"

print("Loading data...")
data = pd.read_csv(TRAIN_DIR + EXTERNAL)  # or CSCADA, depending on which you want to analyze
print(f"Data loaded: {len(data)} rows")
cols = data.columns.tolist()
FEATURE_COLS = [col for col in cols if col not in ["label", "time_window"]]

# X = features, y = labels
X = data[FEATURE_COLS]
y = data["label"]   # attack / benign labels

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance (weights)
weights = rf.feature_importances_

# Normalize weights (optional but recommended)
weights = weights / np.sum(weights)

# Map feature -> weight
feature_weights = dict(zip(FEATURE_COLS, weights))

# for key, value in feature_weights.items():
#     print(f"{key}: {value}")

