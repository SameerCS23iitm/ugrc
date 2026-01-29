# train_iforest.py

import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

from config import FEATURE_COLUMNS, CONTAMINATION
from load_packets import load_packets
from window_aggregate import aggregate_windows

def main():
    packets = load_packets("../data/benign_csv/benign_packets.csv")
    windows = aggregate_windows(packets)

    X = windows[FEATURE_COLUMNS].values

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42
    )

    model.fit(X)

    scores = model.decision_function(X)
    preds = model.predict(X)

    windows["score"] = scores
    windows["anomaly"] = (preds == -1).astype(int)

    windows.to_csv("../results/benign_scores.csv", index=False)
    joblib.dump(model, "../results/iforest.pkl")

    print("Benign anomaly rate:",
          windows["anomaly"].mean())

if __name__ == "__main__":
    main()
