#!/bin/bash

echo "generating benign windows..."
python3 add_iat_b.py
python3 make_windows_b.py
echo "windows generated."

echo "training model..."
python3 train_iforest.py
echo "model trained."

echo "generating attack windows..."
python3 add_iat_a.py
python3 make_windows_a.py
echo "windows generated."

echo "running anomaly detection..."
python3 run_forest.py
echo "anomaly detection complete."