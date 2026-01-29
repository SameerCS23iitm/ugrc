# test_attack.py

import joblib
from config import FEATURE_COLUMNS
from load_packets import load_packets
from window_aggregate import aggregate_windows

def main():
    model = joblib.load("../results/iforest.pkl")

    packets = load_packets("../data/attack_csv/attack_packets.csv")
    windows = aggregate_windows(packets)

    X = windows[FEATURE_COLUMNS].values

    scores = model.decision_function(X)
    preds = model.predict(X)

    windows["score"] = scores
    windows["anomaly"] = (preds == -1).astype(int)

    windows.to_csv("../results/attack_scores.csv", index=False)

    print("Attack anomaly rate:",
          windows["anomaly"].mean())

if __name__ == "__main__":
    main()
