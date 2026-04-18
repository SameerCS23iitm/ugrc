import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TRAIN_DIR = "../train/"
OBS_DIR = "obs/xgboost/"

# Load data
cscada = pd.read_csv(TRAIN_DIR + "labeled_1s_cscada.csv")

FEATURE_COLS = [
    "packet_count", "total_bytes", "mean_packet_size", "std_packet_size",
    "iat_mean", "iat_std", "min_iat", "max_iat",
    "unique_func_codes", "read_count", "write_count", "exception_count"
]

window = 5

# Rolling Std
for col in FEATURE_COLS:
    cscada[f"{col}_rolling_std"] = (
        cscada[col]
        .rolling(window)
        .std()
        .fillna(0)
    )

# Rolling Mean
for col in FEATURE_COLS:
    cscada[f"{col}_rolling_mean"] = (
        cscada[col]
        .rolling(window)
        .mean()
        .fillna(0)
    )

# First Difference
for col in FEATURE_COLS:
    cscada[f"{col}_diff"] = (
        cscada[col]
        .diff()
        .fillna(0)
    )

# Read / Write ratio
cscada["read_write_ratio"] = (
    cscada["read_count"] / (cscada["write_count"] + 1)
)


# Update feature columns
FEATURE_COLS = FEATURE_COLS + [
    col for col in cscada.columns
    if any(
        col.endswith(suffix)
        for suffix in [
            "_rolling_std",
            "_rolling_mean",
            "_diff"
        ]
    )
] + ["read_write_ratio"]

print("Total features:", len(FEATURE_COLS))
print("Example features:", FEATURE_COLS[:10])

# Features and labels from the full dataset
X = cscada[FEATURE_COLS]
y = cscada["label"].to_numpy()


# ================== TRAIN TEST SPLIT ==================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Normalize using training statistics only
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTrain distribution:")
print("Benign:", np.sum(y_train == 0))
print("Attack:", np.sum(y_train == 1))

print("\nTest distribution:")
print("Benign:", np.sum(y_test == 0))
print("Attack:", np.sum(y_test == 1))


# ================== XGBOOST ==================

model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    eval_metric="auc",
    tree_method="hist"
)

model.fit(X_train, y_train)


# ================== EVALUATION ==================

y_prob = model.predict_proba(X_test)[:,1]
y_prob = model.predict_proba(X_test)[:,1]

threshold = 0.2
y_pred = (y_prob > threshold).astype(int)

print("\nResults:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# ================== PLOTS ==================

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax,
    xticklabels=['Benign', 'Attack'],
    yticklabels=['Benign', 'Attack']
)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - XGBoost')
plt.tight_layout()
plt.savefig(OBS_DIR + 'con_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - XGBoost')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'roc.png', dpi=150, bbox_inches='tight')
plt.show()