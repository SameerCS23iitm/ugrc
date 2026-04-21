from rforest import FEATURE_COLS, feature_weights
# from xgb import FEATURE_COLS, feature_weights
# from autoenc_for_weights import FEATURE_COLS, feature_weights
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import numpy as np

SEQ_LEN = 20
TRAIN_DIR = "../train/"
DATA_DIR = "../data/"

# Keep average feature scale near 1.0 so weighted inputs do not collapse in magnitude.
FEATURE_WEIGHT_VEC = np.array([feature_weights[col] for col in FEATURE_COLS], dtype=np.float32)
# FEATURE_WEIGHT_VEC = FEATURE_WEIGHT_VEC / FEATURE_WEIGHT_VEC.mean()

for gran in ["1s"]:
    df = pd.read_csv(TRAIN_DIR + f"labeled_{gran}_cscada.csv")
    print(f"\n{gran} cscada:")
    print(f"  Total : {len(df)}")
    print(f"  Attack: {df['label'].sum()} ({100*df['label'].sum()/len(df):.1f}%)")

def build_sequences(df, seq_len=SEQ_LEN):
    X, y = [], []
    
    for (src, dst), group in df.groupby(["ip.src", "ip.dst"]):
        group = group.sort_values("time_window").reset_index(drop=True)
        features = group[FEATURE_COLS].values.astype(np.float32)
        labels   = group["label"].values
        
        if len(group) < seq_len:
            continue
            
        for i in range(len(group) - seq_len):
            X.append(features[i:i+seq_len])
            y.append(labels[i+seq_len]) #check this labeling once, might wanna change this
    
    return np.array(X), np.array(y)

for gran in ["1s"]:
    print(f"\nBuilding sequences for {gran}...")
    df = pd.read_csv(TRAIN_DIR + f"labeled_{gran}_cscada.csv")
    
    X, y = build_sequences(df)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Attack sequences: {y.sum()} ({100*y.sum()/len(y):.1f}%)")
    
    np.save(TRAIN_DIR + f"X_{gran}.npy", X)
    np.save(TRAIN_DIR + f"y_{gran}.npy", y)
    print(f"  Saved → X_{gran}.npy, y_{gran}.npy")

for gran in ["1s"]:
    df = pd.read_csv(TRAIN_DIR + f"labeled_{gran}_cscada.csv")
    
    scaler = MinMaxScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    joblib.dump(scaler, DATA_DIR + f"scaler_{gran}.pkl")
    
    # Rebuild sequences
    X, y = build_sequences(df)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    np.save(TRAIN_DIR + f"X_train_{gran}.npy", X_train)
    np.save(TRAIN_DIR + f"X_test_{gran}.npy",  X_test)
    np.save(TRAIN_DIR + f"y_train_{gran}.npy", y_train)
    np.save(TRAIN_DIR + f"y_test_{gran}.npy",  y_test)
    
    print(f"{gran}: X_train {X_train.shape} | attacks in train: {y_train.sum()} | attacks in test: {y_test.sum()}")

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.layers import Input, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import seaborn as sns

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Load 1-second windowed data
X_train = np.load('../train/X_train_1s.npy')
X_test = np.load('../train/X_test_1s.npy')
y_train = np.load('../train/y_train_1s.npy')
y_test = np.load('../train/y_test_1s.npy')

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"\nClass distribution (train):")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count} samples ({100*count/len(y_train):.1f}%)")
print(f"\nClass distribution (test):")
unique, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count} samples ({100*count/len(y_test):.1f}%)")

print(f"✓ Data is ready for LSTM:")
print(f"  X_train: {X_train.shape} = (samples={X_train.shape[0]}, timesteps={X_train.shape[1]}, features={X_train.shape[2]})")
print(f"  X_test: {X_test.shape} = (samples={X_test.shape[0]}, timesteps={X_test.shape[1]}, features={X_test.shape[2]})")


# Initialize learnable weights from autoencoder
init_weights = FEATURE_WEIGHT_VEC.reshape(1, 1, -1)

inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Learnable feature weights
feature_weights = tf.Variable(
    initial_value=init_weights,
    trainable=True,
    dtype=tf.float32,
    name="feature_weights"
)

weighted = Multiply()([inputs, feature_weights])

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

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

print(f"\n✓ Training complete! Stopped at epoch {len(history.history['loss'])}")

# Get predictions
y_pred_prob = model.predict(X_test).flatten()

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]

print("Best threshold:", best_threshold)

y_pred = (y_pred_prob > best_threshold).astype(int)

# Metrics
print("="*60)
print("TEST SET RESULTS")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['Benign', 'Attack'], 
            yticklabels=['Benign', 'Attack'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - LSTM Baseline')
plt.tight_layout()
plt.savefig('../observations/confusion_matrix_lstm.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Confusion matrix saved to observations/")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'LSTM (AUC = {roc_auc_score(y_test, y_pred_prob):.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - LSTM Baseline')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../observations/roc_curve_lstm.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"ROC curve saved to observations/")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Loss
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy
ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../observations/training_history_lstm.png', dpi=150, bbox_inches='tight')
plt.show()

model.save('../train/lstm_baseline_1s.keras')
print("✓ Model saved to train/lstm_baseline_1s.keras")