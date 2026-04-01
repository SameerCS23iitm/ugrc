#!/usr/bin/env python3
"""
Diagnostic script to understand data shape and reshape issues
"""
import numpy as np
import traceback

print("=" * 70)
print("DATA SHAPE DIAGNOSIS")
print("=" * 70)

try:
    import os
    print("\n1. Loading .npy files...")
    print(f"   Current working directory: {os.getcwd()}")
    X_train = np.load('./train/X_train_1s.npy')
    X_test = np.load('./train/X_test_1s.npy')
    y_train = np.load('./train/y_train_1s.npy')
    y_test = np.load('./train/y_test_1s.npy')
    print("   ✓ Loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading: {e}")
    traceback.print_exc()
    exit(1)

print("\n2. Original shapes:")
print(f"   X_train: {X_train.shape} | dtype: {X_train.dtype}")
print(f"   X_test:  {X_test.shape} | dtype: {X_test.dtype}")
print(f"   y_train: {y_train.shape} | dtype: {y_train.dtype}")
print(f"   y_test:  {y_test.shape} | dtype: {y_test.dtype}")

print("\n3. Checking dimensions:")
print(f"   X_train.ndim: {X_train.ndim}")
print(f"   X_test.ndim:  {X_test.ndim}")

print("\n4. Attempting reshape to 3D [samples, timesteps, features=1]:")
try:
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    print(f"   ✓ X_train reshape OK → {X_train_reshaped.shape}")
except Exception as e:
    print(f"   ✗ X_train reshape FAILED")
    print(f"   Error: {e}")
    print(f"   Trying to reshape {X_train.shape} → ({X_train.shape[0]}, {X_train.shape[1]}, 1)")
    print(f"   Total elements: {np.prod(X_train.shape)} → {X_train.shape[0] * X_train.shape[1] * 1}")
    traceback.print_exc()

try:
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print(f"   ✓ X_test reshape OK → {X_test_reshaped.shape}")
except Exception as e:
    print(f"   ✗ X_test reshape FAILED")
    print(f"   Error: {e}")
    traceback.print_exc()

print("\n5. Alternative: Flatten and explore:")
print(f"   X_train first 3 values: {X_train.flatten()[:3]}")
print(f"   X_train last 3 values: {X_train.flatten()[-3:]}")

print("\n6. CSV alternative (if .npy is problematic):")
try:
    import pandas as pd
    df = pd.read_csv('./train/labeled_1s_benign.csv')
    print(f"   ✓ labeled_1s_benign.csv loaded: shape={df.shape}")
    print(f"   Columns: {list(df.columns)[:5]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("If reshape still fails, check:")
print("  • Are X_*.npy files from a previous run?")
print("  • Do they have the right dimensions?")
print("  • Should we use CSVs instead?")
print("=" * 70)
