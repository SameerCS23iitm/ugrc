import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("../data/train/benign_windows.csv")
df = df.dropna()

feature_names = df.columns.tolist()
X = df.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_scaled)

# Save everything needed for runtime
joblib.dump(feature_names, "../data/models/feature_names.pkl")
joblib.dump(model, "../data/models/iforest_benign.pkl")
joblib.dump(scaler, "../data/models/scaler.pkl")

scores = model.decision_function(X_scaled)
preds = model.predict(X_scaled)

print("Model trained on benign data")
print("Anomalies in benign:", (preds == -1).sum(), "/", len(preds))
print("Score range:", scores.min(), scores.max())
