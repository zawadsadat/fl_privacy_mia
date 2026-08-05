"""
Data loader for the Heart Disease dataset (UCI).

Combined dataset from four medical centers:
  Cleveland (303), Hungarian (294), Switzerland (123), VA Long Beach (200)
  Total: ~920 samples, 13 features

Binary classification: presence (1) vs absence (0) of heart disease.
Original target has values 0-4; we binarize to 0 vs 1+.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

UCI_URLS = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "Switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "VA Long Beach": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}


def _load_single_center(url, center_name):
    """Load a single center's data from UCI."""
    try:
        df = pd.read_csv(url, names=COLUMNS, na_values="?", header=None)
        df["center"] = center_name
        print(f"  Loaded {center_name}: {len(df)} samples")
        return df
    except Exception as e:
        print(f"  WARNING: Could not load {center_name}: {e}")
        return None


def load_heart_disease():
    """
    Load and preprocess the combined Heart Disease dataset (all 4 centers).

    Returns:
        X (np.ndarray): Feature matrix, standardized.
        y (np.ndarray): Binary labels (0 = no disease, 1 = disease).
    """
    print("Loading Heart Disease dataset (4 centers)...")

    # Download from UCI
    dfs = []
    for center, url in UCI_URLS.items():
        df = _load_single_center(url, center)
        if df is not None:
            dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError("Could not load Heart Disease dataset from UCI.")

    # Concatenate all centers
    df = pd.concat(dfs, ignore_index=True)
    center_info = df["center"].value_counts()
    df = df.drop(columns=["center"])

    # Convert all to numeric
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where target is missing (can't impute labels)
    n_before = len(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing target")

    # Separate features and target BEFORE imputation
    y_raw = df["target"].values
    X_df = df.drop(columns=["target"])

    # Impute missing feature values with median (per column)
    n_missing = X_df.isna().sum().sum()
    if n_missing > 0:
        print(f"  Imputing {n_missing} missing feature values with column median")
        X_df = X_df.fillna(X_df.median())

    X = X_df.values.astype(np.float32)

    # Binarize target: 0 = no disease, 1+ = disease
    y = (y_raw > 0).astype(np.float32)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"\nHeart Disease (combined): {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.1%}")
    print(f"  Centers loaded:")
    for center, count in center_info.items():
        print(f"    {center}: {count}")

    return X.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    X, y = load_heart_disease()
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
