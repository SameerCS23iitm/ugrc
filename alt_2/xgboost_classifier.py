import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier

TRAIN_DIR = "../train/"
OBS_DIR = "obs/xgboost/"

# Load data
cscada = pd.read_csv(TRAIN_DIR + "labeled_1s_cscada.csv")
cols = cscada.columns.tolist()
BASE_FEATURE_COLS = [col for col in cols if col not in ["label", "time_window"]]

window = 5

def add_rolling_features(df, feature_cols, rolling_window):
    df = df.sort_values("time_window").reset_index(drop=True).copy()

    for col in feature_cols:
        df[f"{col}_rolling_std"] = (
            df[col]
            .rolling(rolling_window)
            .std()
            .fillna(0)
        )

    for col in feature_cols:
        df[f"{col}_rolling_mean"] = (
            df[col]
            .rolling(rolling_window)
            .mean()
            .fillna(0)
        )

    for col in feature_cols:
        df[f"{col}_diff"] = (
            df[col]
            .diff()
            .fillna(0)
        )

    feature_cols = feature_cols + [
        col for col in df.columns
        if any(
            col.endswith(suffix)
            for suffix in [
                "_rolling_std",
                "_rolling_mean",
                "_diff"
            ]
        )
    ]

    return df, feature_cols


def temporal_disjoint_split(df, train_ratio=0.8):
    df = df.sort_values("time_window").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df

print("Base features:", len(BASE_FEATURE_COLS))
print("Example features:", BASE_FEATURE_COLS[:10])

train_raw, test_raw = temporal_disjoint_split(cscada)
train_df, FEATURE_COLS = add_rolling_features(train_raw, BASE_FEATURE_COLS, window)
test_df, _ = add_rolling_features(test_raw, BASE_FEATURE_COLS, window)

print("Total features:", len(FEATURE_COLS))

print("Disjoint split sizes:")
print("Train rows:", len(train_df))
print("Test rows:", len(test_df))

X_train = train_df[FEATURE_COLS]
y_train = train_df["label"].to_numpy()
X_test = test_df[FEATURE_COLS]
y_test = test_df["label"].to_numpy()


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