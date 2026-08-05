"""
Data loader router.
Loads the correct dataset based on config.DATASET.
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.config import DATASET, TEST_SIZE


def get_raw_data():
    """
    Load and return (X, y) for the configured dataset.
    All datasets return standardized features and binary float32 labels.
    """
    if DATASET == "breast_cancer":
        return _load_breast_cancer()
    elif DATASET == "heart_disease":
        from utils.data_loader_heart_disease import load_heart_disease
        return load_heart_disease()
    elif DATASET == "cardiotocography":
        from utils.data_loader_cardiotocography import load_cardiotocography
        return load_cardiotocography()
    elif DATASET == "diabetes_hospital":
        from utils.data_loader_diabetes_hospital import load_diabetes_hospital
        return load_diabetes_hospital()
    else:
        raise ValueError(f"Unknown dataset: {DATASET}. "
                         f"Options: breast_cancer, heart_disease, cardiotocography, diabetes_hospital")


def _load_breast_cancer():
    """Load Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"Breast Cancer Wisconsin: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.1%}")

    return X.astype(np.float32), y.astype(np.float32)


def load_data(test_size=None, random_state=42):
    """
    Load the active dataset as torch tensors.
    Returns:
        X_train, X_test, y_train, y_test as float32 tensors
    """
    if test_size is None:
        test_size = TEST_SIZE
    X, y = get_raw_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


def get_input_dim():
    """Return the feature dimensionality of the active dataset."""
    X, _ = get_raw_data()
    return X.shape[1]


def get_dataset_info():
    """Return a dict with dataset metadata for logging."""
    X, y = get_raw_data()
    return {
        "name": DATASET,
        "samples": len(y),
        "features": X.shape[1],
        "positive_rate": y.mean(),
    }


if __name__ == "__main__":
    X, y = get_raw_data()
    print(f"\nDataset: {DATASET}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"TEST_SIZE: {TEST_SIZE}")
    print(f"Train samples: ~{int(len(X) * (1 - TEST_SIZE))}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
