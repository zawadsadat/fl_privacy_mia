"""
Privacy Budget (ε) Sweep.

Varies noise_multiplier to measure the full privacy-utility-vulnerability
tradeoff curve. For each noise level:
  1. Train target model with DP-SGD
  2. Train shadow models with matching DP
  3. Run MIA (Random Forest)
  4. Record ε, test accuracy, AUC, TPR, loss gap

Usage:
    python experiments/run_epsilon_sweep.py
    python experiments/run_epsilon_sweep.py --noise-multipliers 0.5 1.0 2.0
"""

import sys
import os
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from opacus import PrivacyEngine

from models.model import TargetModel
from utils.data_loader import get_raw_data, get_input_dim, get_dataset_info
from utils.config import (
    LR, BATCH_SIZE, TEST_SIZE, MAX_GRAD_NORM,
    SHADOW_EPOCHS_DP, NUM_SHADOW_MODELS,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_dp(X_train, y_train, input_dim, noise_multiplier, epochs):
    """Train a model with DP-SGD at a given noise level."""
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TargetModel(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    return model, epsilon


def train_no_dp(X_train, y_train, input_dim, epochs):
    """Train without DP (ε = ∞ baseline)."""
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TargetModel(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    return model


def extract_attack_features(model, X, y):
    """Extract [prob0, prob1, loss, entropy] per sample."""
    features = []
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_t).squeeze(-1)
        for i, p in enumerate(preds):
            c = p.item()
            loss = F.binary_cross_entropy(p, y_t[i], reduction='none').item()
            entropy = -(c * math.log(c + 1e-10) + (1-c) * math.log(1-c + 1e-10))
            features.append([1-c, c, loss, entropy])

    return np.array(features)


def run_mia(shadow_features_list, shadow_labels_list):
    """Train RF attack and return metrics."""
    X_attack = np.vstack(shadow_features_list)
    y_attack = np.concatenate(shadow_labels_list)

    # Balance
    member_idx = np.where(y_attack == 1)[0]
    nonmember_idx = np.where(y_attack == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    m_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nm_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)
    balanced_idx = np.concatenate([m_sample, nm_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = X_attack[balanced_idx], y_attack[balanced_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight='balanced'
    )
    rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, probs)
    bal_acc = balanced_accuracy_score(y_te, (probs > 0.5).astype(float))

    fpr, tpr, _ = roc_curve(y_te, probs)
    tpr_at = {}
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at[target_fpr] = tpr[idx]

    # Loss gap
    member_mask = y_attack == 1
    loss_gap = abs(X_attack[member_mask, 2].mean() - X_attack[~member_mask, 2].mean())

    return {
        "auc": auc, "bal_acc": bal_acc,
        "tpr_1": tpr_at[0.01], "tpr_5": tpr_at[0.05], "tpr_10": tpr_at[0.10],
        "loss_gap": loss_gap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-multipliers", nargs="+", type=float,
                        default=[0.3, 0.5, 0.8, 1.0, 1.3, 2.0, 3.0, 5.0])
    parser.add_argument("--num-shadow", type=int, default=10,
                        help="Shadow models per noise level (fewer = faster)")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    info = get_dataset_info()
    input_dim = get_input_dim()

    print(f"{'=' * 70}")
    print(f"  Privacy Budget (ε) Sweep")
    print(f"  Dataset: {info['name']} ({info['samples']} samples)")
    print(f"  Noise multipliers: {args.noise_multipliers}")
    print(f"  Shadow models per config: {args.num_shadow}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'=' * 70}\n")

    X_all, y_all = get_raw_data()

    # --- No-DP baseline first ---
    print("Training no-DP baseline...")
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=42, stratify=y_all
    )

    nodp_model = train_no_dp(X_train_base, y_train_base, input_dim, args.epochs)

    with torch.no_grad():
        X_te_t = torch.tensor(X_test_base, dtype=torch.float32).to(device)
        y_te_t = torch.tensor(y_test_base, dtype=torch.float32).to(device)
        preds = nodp_model(X_te_t).squeeze(-1)
        nodp_acc = ((preds > 0.5).float() == y_te_t).float().mean().item()

    # Shadow models for no-DP
    print("Training no-DP shadow models...")
    nodp_features = []
    nodp_labels = []

    for s in range(args.num_shadow):
        X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, random_state=s, stratify=y_all
        )
        shadow = train_no_dp(X_s_tr, y_s_tr, input_dim, 30)

        mem_feat = extract_attack_features(shadow, X_s_tr, y_s_tr)
        nonmem_feat = extract_attack_features(shadow, X_s_te, y_s_te)

        nodp_features.append(mem_feat)
        nodp_labels.append(np.ones(len(mem_feat)))
        nodp_features.append(nonmem_feat)
        nodp_labels.append(np.zeros(len(nonmem_feat)))

        print(f"  Shadow {s+1}/{args.num_shadow} done")

    nodp_mia = run_mia(nodp_features, nodp_labels)

    print(f"\n  No-DP: TestAcc={nodp_acc:.4f}, MIA AUC={nodp_mia['auc']:.4f}, "
          f"LossGap={nodp_mia['loss_gap']:.4f}")

    all_results = []

    # --- Sweep noise multipliers ---
    for nm in args.noise_multipliers:
        print(f"\n{'─' * 70}")
        print(f"  noise_multiplier = {nm}")
        print(f"{'─' * 70}")

        start = time.time()

        # Train target model
        target, epsilon = train_model_dp(
            X_train_base, y_train_base, input_dim, nm, args.epochs
        )

        with torch.no_grad():
            preds = target(X_te_t).squeeze(-1)
            test_acc = ((preds > 0.5).float() == y_te_t).float().mean().item()

        # Train shadow models with same noise
        shadow_features = []
        shadow_labels = []

        for s in range(args.num_shadow):
            X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
                X_all, y_all, test_size=TEST_SIZE, random_state=s, stratify=y_all
            )

            shadow, _ = train_model_dp(X_s_tr, y_s_tr, input_dim, nm, args.epochs)

            mem_feat = extract_attack_features(shadow, X_s_tr, y_s_tr)
            nonmem_feat = extract_attack_features(shadow, X_s_te, y_s_te)

            shadow_features.append(mem_feat)
            shadow_labels.append(np.ones(len(mem_feat)))
            shadow_features.append(nonmem_feat)
            shadow_labels.append(np.zeros(len(nonmem_feat)))

            print(f"  Shadow {s+1}/{args.num_shadow} done")

        mia = run_mia(shadow_features, shadow_labels)
        elapsed = time.time() - start

        result = {
            "noise_multiplier": nm,
            "epsilon": epsilon,
            "test_acc": test_acc,
            **mia,
            "time_sec": elapsed,
        }
        all_results.append(result)

        print(f"\n  ε={epsilon:.2f}, TestAcc={test_acc:.4f}, "
              f"MIA AUC={mia['auc']:.4f}, LossGap={mia['loss_gap']:.4f} "
              f"({elapsed:.0f}s)")

    # --- Summary ---
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY: Privacy Budget Sweep")
    print(f"{'=' * 80}")
    print(f"  {'NoiseMul':>8} {'ε':>8} {'TestAcc':>8} {'AUC':>8} {'BalAcc':>8} "
          f"{'T@1%':>7} {'T@5%':>7} {'T@10%':>7} {'LossGap':>8}")
    print(f"  {'-' * 73}")

    # No-DP row
    print(f"  {'∞ (none)':>8} {'∞':>8} {nodp_acc:>8.4f} {nodp_mia['auc']:>8.4f} "
          f"{nodp_mia['bal_acc']:>8.4f} {nodp_mia['tpr_1']:>7.4f} "
          f"{nodp_mia['tpr_5']:>7.4f} {nodp_mia['tpr_10']:>7.4f} "
          f"{nodp_mia['loss_gap']:>8.4f}")

    for r in sorted(all_results, key=lambda x: -x["epsilon"]):
        print(f"  {r['noise_multiplier']:>8.1f} {r['epsilon']:>8.2f} "
              f"{r['test_acc']:>8.4f} {r['auc']:>8.4f} {r['bal_acc']:>8.4f} "
              f"{r['tpr_1']:>7.4f} {r['tpr_5']:>7.4f} {r['tpr_10']:>7.4f} "
              f"{r['loss_gap']:>8.4f}")

    print(f"  {'-' * 73}")

    # --- Save ---
    os.makedirs("experiments", exist_ok=True)
    np.save("experiments/epsilon_sweep_results.npy", {
        "nodp": {"test_acc": nodp_acc, **nodp_mia},
        "sweep": all_results,
    }, allow_pickle=True)

    print(f"\nResults saved to experiments/epsilon_sweep_results.npy")

    # --- Save as CSV for audit score ---
    import pandas as pd
    csv_rows = [{"NoiseMul": float("inf"), "epsilon": float("inf"),
                 "TestAcc": nodp_acc, "AUC": nodp_mia["auc"],
                 "BalAcc": nodp_mia["bal_acc"], "T@1%": nodp_mia["tpr_1"],
                 "T@5%": nodp_mia["tpr_5"], "T@10%": nodp_mia["tpr_10"],
                 "LossGap": nodp_mia["loss_gap"]}]
    for r in sorted(all_results, key=lambda x: -x["epsilon"]):
        csv_rows.append({"NoiseMul": r["noise_multiplier"], "epsilon": r["epsilon"],
                         "TestAcc": r["test_acc"], "AUC": r["auc"],
                         "BalAcc": r["bal_acc"], "T@1%": r["tpr_1"],
                         "T@5%": r["tpr_5"], "T@10%": r["tpr_10"],
                         "LossGap": r["loss_gap"]})
    pd.DataFrame(csv_rows).to_csv("experiments/epsilon_sweep_results.csv", index=False)
    print(f"Results saved to experiments/epsilon_sweep_results.csv")


if __name__ == "__main__":
    main()
