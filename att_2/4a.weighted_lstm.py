from rforest import FEATURE_COLS, feature_weights
# from xgb import FEATURE_COLS, feature_weights
# from autoenc_for_weights import FEATURE_COLS, feature_weights

import os
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from tensorflow.keras.layers import Input, Multiply, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN    = 20
BATCH_SIZE = 16          # halved from 32 to ease memory pressure
TRAIN_DIR  = "../train/"
DATA_DIR   = "../data/"
OBS_DIR    = "ext/rforest/"

FEATURE_WEIGHT_VEC = np.array(
    [feature_weights[col] for col in FEATURE_COLS], dtype=np.float32
)

# ─────────────────────────────────────────────
#  LIMIT TF/GPU MEMORY GROWTH
#    Prevents TensorFlow from grabbing all VRAM
#    at startup. Safe even with no GPU.
# ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ─────────────────────────────────────────────
#  QUICK DATASET STATS (no sequences yet)
# ─────────────────────────────────────────────
gran = "1s"
df = pd.read_csv(TRAIN_DIR + f"labeled_{gran}_external.csv")
print(f"\n{gran} external:")
print(f"  Total : {len(df)}")
print(f"  Attack: {df['label'].sum()} ({100*df['label'].sum()/len(df):.1f}%)")

# ─────────────────────────────────────────────
#  SCALE FEATURES  (fit on full set, saved to disk)
#    We only store the scaler and the scaled
#    feature matrix — NOT the sequences.
# ─────────────────────────────────────────────
df = df.sort_values("time_window").reset_index(drop=True)

scaler = MinMaxScaler()
df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS]).astype(np.float32)
joblib.dump(scaler, DATA_DIR + f"scaler_{gran}.pkl")
print(f"\nScaler saved → {DATA_DIR}scaler_{gran}.pkl")

# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT
#
#    Where are the sequences?
#    ─────────────────────────────────────────
#    They are NEVER fully built. The generator
#    below slices them on-the-fly per batch.
#
#    train_gen  →  rows 0 .. split+SEQ_LEN
#    test_gen   →  rows split .. end
#
#    The +SEQ_LEN overlap ensures the generator
#    can form a full window at every index right
#    up to the split boundary.
#
#    y_test is kept as a plain array because it
#    is just integers — tiny, no memory issue.
# ─────────────────────────────────────────────
features = df[FEATURE_COLS].values.astype(np.float32)
labels   = df["label"].values

n_windows = len(features) - SEQ_LEN          # total possible sequence starts
split     = int(n_windows * 0.8)             # 80 % train, 20 % test

# y_test: labels for the test windows
# (window i predicts labels[i + SEQ_LEN])
y_test = labels[split + SEQ_LEN:]

del df   # free the DataFrame now that we have the arrays
gc.collect()

print(f"\nTotal windows : {n_windows}")
print(f"Train windows : {split}")
print(f"Test  windows : {n_windows - split}")
print(f"Attack % train: {100 * labels[:split+SEQ_LEN].sum() / (split+SEQ_LEN):.1f}%")
print(f"Attack % test : {100 * y_test.sum() / len(y_test):.1f}%")

# ─────────────────────────────────────────────
#  SEQUENCE GENERATOR
#    Builds one batch of (BATCH_SIZE, SEQ_LEN, n_features)
#    on every __getitem__ call. Peak RAM = one batch.
# ─────────────────────────────────────────────
class SequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, features, labels, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
        self.features   = features          # already float32
        self.labels     = labels
        self.seq_len    = seq_len
        self.batch_size = batch_size
        # valid start indices: we need i+seq_len to be a valid label index
        self.indices    = np.arange(len(features) - seq_len)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = np.stack([self.features[i : i + self.seq_len] for i in batch])
        y = np.array([self.labels[i + self.seq_len]       for i in batch])
        return X, y

# train slice: indices 0 .. split  (needs features up to split+SEQ_LEN)
train_gen = SequenceGenerator(
    features[: split + SEQ_LEN],
    labels  [: split + SEQ_LEN]
)

# test slice: indices 0 .. (n_windows-split)  mapped to the test portion
test_gen = SequenceGenerator(
    features[split :],
    labels  [split :]
)

# ─────────────────────────────────────────────
#  CLASS WEIGHTS  (computed from label counts,
#    no sequence arrays needed)
# ─────────────────────────────────────────────
train_labels = labels[: split + SEQ_LEN]
weights      = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_labels
)
class_weights = {0: weights[0], 1: weights[1]}
del train_labels
gc.collect()

print(f"\nClass weights → benign: {class_weights[0]:.3f}, attack: {class_weights[1]:.3f}")

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
n_features   = len(FEATURE_COLS)
init_weights = FEATURE_WEIGHT_VEC.reshape(1, 1, -1)

inputs = Input(shape=(SEQ_LEN, n_features))

learnable_weights = tf.Variable(
    initial_value=init_weights,
    trainable=True,
    dtype=tf.float32,
    name="feature_weights"
)

weighted = Multiply()([inputs, learnable_weights])

x = LSTM(64)(weighted)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(model.summary())

# ─────────────────────────────────────────────
#  TRAIN
#    validation_data=test_gen  (not validation_split —
#    that only works with raw arrays, not generators)
# ─────────────────────────────────────────────
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=test_gen,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

print(f"\nTraining complete — stopped at epoch {len(history.history['loss'])}")

# ─────────────────────────────────────────────
# ⑨ EVALUATE
# ─────────────────────────────────────────────
y_pred_prob = model.predict(test_gen).flatten()
y_test = y_test[:len(y_pred_prob)]

# Precision-recall curve & best threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores   = 2 * precision * recall / (precision + recall + 1e-8)
best_idx    = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold (max F1): {best_threshold:.4f}")

y_pred = (y_pred_prob > best_threshold).astype(int)

print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

# ─────────────────────────────────────────────
# ⑩ PLOTS
# ─────────────────────────────────────────────
# Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(OBS_DIR + 'pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion matrix
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Benign', 'Attack'],
            yticklabels=['Benign', 'Attack'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - LSTM')
plt.tight_layout()
plt.savefig(OBS_DIR + 'con_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2,
        label=f'LSTM (AUC = {roc_auc_score(y_test, y_pred_prob):.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - LSTM')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OBS_DIR + 'roc.png', dpi=150, bbox_inches='tight')
plt.show()

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(history.history['loss'],     label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(history.history['accuracy'],     label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OBS_DIR + 'losses.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
model.save(TRAIN_DIR + 'lstm_baseline_1s.keras')
print("Model saved → train/lstm_baseline_1s.keras")

os.system('notify-send "Python Script" "Execution complete!"')