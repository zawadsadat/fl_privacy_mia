import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.config import TEST_SIZE as DEFAULT_TEST_SIZE


def load_data(test_size=None, random_state=42):
    if test_size is None:
        test_size = DEFAULT_TEST_SIZE

    data = load_breast_cancer()

    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


def get_raw_data():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# Backward compatibility
def load_diabetes_data(path=None):
    """Deprecated: use load_data() instead."""
    return load_data()
