#!/bin/bash

echo "training iforest..."
python3 train_iforest.py
echo "iforest trained."
echo "running on attack data..."
python3 test_attack.py
echo "done."