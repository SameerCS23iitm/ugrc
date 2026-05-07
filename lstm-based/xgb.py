from xgboost import XGBClassifier
import numpy as np
import pandas as pd

TRAIN_DIR = "../train/"
CSCADA = "labeled_1s_cscada.csv"
EXTERNAL = "labeled_1s_external.csv"

data = pd.read_csv(TRAIN_DIR + EXTERNAL)  # or CSCADA, depending on which you want to analyze

cols = data.columns.tolist()
FEATURE_COLS = [col for col in cols if col not in ["label", "time_window"]]


# Features and labels
# X = features, y = labels
X = data[FEATURE_COLS]
y = data["label"]   # assume 1 = attack, 0 = benign
print(f"Attack count: {y.sum()}, Benign count: {len(y) - y.sum()}")

# Handle imbalance
scale_pos_weight = (len(y) - y.sum()) / y.sum()

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X, y)

# Get feature importance (weights)
weights = model.feature_importances_

# Normalize
weights = weights / np.sum(weights)

feature_weights = dict(zip(FEATURE_COLS, weights))

# for key, value in feature_weights.items():
#     print(f"{key}: {value}")
