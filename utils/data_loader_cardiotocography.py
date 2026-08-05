"""
Data loader for the Cardiotocography (CTG) dataset (UCI).

2,126 fetal cardiotocograms with 21 features.
Binary classification: Normal vs Suspect+Pathologic.

Features are extracted from fetal heart rate and uterine contraction
signals, used for fetal monitoring during pregnancy.

Source: https://archive.ics.uci.edu/dataset/193/cardiotocography
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_cardiotocography():
    """
    Load and preprocess the Cardiotocography dataset.

    Returns:
        X (np.ndarray): Feature matrix, standardized.
        y (np.ndarray): Binary labels (0 = normal, 1 = suspect/pathologic).
    """
    from sklearn.datasets import fetch_openml

    try:
        data = fetch_openml(name="cardiotocography", version=1, as_frame=True, parser="auto")
        df = data.frame
    except Exception:
        try:
            data = fetch_openml(data_id=1560, as_frame=True, parser="auto")
            df = data.frame
        except Exception as e:
            raise RuntimeError(f"Could not load Cardiotocography dataset: {e}")

    print(f"  Raw columns ({len(df.columns)}): {list(df.columns)}")

    # Find the NSP column (3-class: Normal/Suspect/Pathologic)
    nsp_col = None
    for col_name in ["NSP", "nsp"]:
        if col_name in df.columns:
            nsp_col = col_name
            break

    if nsp_col is None:
        # Try last column
        nsp_col = df.columns[-1]
        print(f"  WARNING: NSP column not found, using last column: {nsp_col}")

    # Separate target
    y_raw = pd.to_numeric(df[nsp_col], errors="coerce")
    X_df = df.drop(columns=[nsp_col])

    # Drop CLASS column (10-class label) if present — we only use NSP
    for col in ["CLASS", "class", "Class"]:
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])
            print(f"  Dropped {col} column (10-class label, not used)")

    # Convert all features to numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    mask = X_df.notna().all(axis=1) & y_raw.notna()
    X_df = X_df[mask].reset_index(drop=True)
    y_raw = y_raw[mask].reset_index(drop=True)

    # Check unique target values to determine encoding
    unique_vals = sorted(y_raw.unique())
    print(f"  Target unique values: {unique_vals}")

    # Determine binarization based on actual values
    if set(unique_vals).issubset({1, 2, 3}) or set(unique_vals).issubset({1.0, 2.0, 3.0}):
        # Standard encoding: 1=Normal, 2=Suspect, 3=Pathologic
        # Binary: Normal(1) = 0, Suspect+Pathologic(2,3) = 1
        y = (y_raw > 1).astype(np.float32).values
        print(f"  Binarization: Normal(1)=0, Suspect+Pathologic(2,3)=1")
    elif set(unique_vals).issubset({"N", "S", "P"}) or set(unique_vals).issubset({"Normal", "Suspect", "Pathologic"}):
        # String encoding
        y = np.where(y_raw.isin(["N", "Normal"]), 0.0, 1.0).astype(np.float32)
        print(f"  Binarization: Normal=0, Suspect+Pathologic=1")
    else:
        # Unknown encoding — check if majority class is the first unique value
        print(f"  WARNING: Unknown target encoding. Values: {unique_vals}")
        # Count each value
        val_counts = y_raw.value_counts()
        print(f"  Value counts:\n{val_counts}")
        # Assume most frequent = Normal = 0
        majority_val = val_counts.index[0]
        y = (y_raw != majority_val).astype(np.float32).values
        print(f"  Treating {majority_val} as Normal(0), rest as Abnormal(1)")

    X = X_df.values.astype(np.float32)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Verify positive rate is reasonable (should be ~22% for Suspect+Pathologic)
    pos_rate = y.mean()
    if pos_rate > 0.5:
        print(f"  WARNING: Positive rate is {pos_rate:.1%}, which seems inverted. Flipping labels.")
        y = 1.0 - y
        pos_rate = y.mean()

    print(f"\nCardiotocography: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate (suspect/pathologic): {pos_rate:.1%}")
    print(f"  Domain: fetal heart rate monitoring")

    return X.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    X, y = load_cardiotocography()
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
