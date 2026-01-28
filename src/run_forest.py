import pandas as pd
import joblib

model = joblib.load("../data/models/iforest_benign.pkl")
scaler = joblib.load("../data/models/scaler.pkl")
feature_names = joblib.load("../data/models/feature_names.pkl")

df_attack = pd.read_csv("../data/test/attack_windows.csv")
df_attack = df_attack[feature_names]
df_attack = df_attack.dropna()

X_attack = df_attack.values
X_attack_scaled = scaler.transform(X_attack)

attack_scores = model.decision_function(X_attack_scaled)
attack_preds = model.predict(X_attack_scaled)

print("Anomalies in attack data:")
print((attack_preds == -1).sum(), "out of", len(attack_preds))
print("Attack score range:", attack_scores.min(), attack_scores.max())
