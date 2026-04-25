import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from models.model import TargetModel
from utils.data_loader import get_raw_data
from utils.config import (
    NUM_SHADOW_MODELS, SHADOW_EPOCHS_NO_DP, SHADOW_EPOCHS_DP,
    LR, BATCH_SIZE, TEST_SIZE, INPUT_DIM,
    NOISE_MULTIPLIER, MAX_GRAD_NORM,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS_NO_DP = SHADOW_EPOCHS_NO_DP
EPOCHS_DP = SHADOW_EPOCHS_DP


def extract_attack_features(model, X, y):
    
    features = []

    with torch.no_grad():
        preds = model(X).squeeze()

        for i, p in enumerate(preds):
            confidence = p.item()
            prob0 = 1 - confidence
            prob1 = confidence

            sample_loss = F.binary_cross_entropy(
                p, y[i], reduction='none'
            ).item()

            entropy = -(
                confidence * math.log(confidence + 1e-10) +
                (1 - confidence) * math.log(1 - confidence + 1e-10)
            )

            features.append([prob0, prob1, sample_loss, entropy])

    return features


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dp", action="store_true", help="Train with DP-SGD")
    group.add_argument("--no-dp", action="store_true", help="Train without DP")
    args = parser.parse_args()

    use_dp = args.dp
    epochs = EPOCHS_DP if use_dp else EPOCHS_NO_DP
    mode_str = "WITH DP-SGD" if use_dp else "WITHOUT DP (baseline)"
    suffix = "_dp" if use_dp else "_nodp"

    print(f"=== Shadow Model Training {mode_str} ===\n")

    X_all, y_all = get_raw_data()

    attack_X = []
    attack_y = []

    for model_id in range(NUM_SHADOW_MODELS):

        print(f"Training Shadow Model {model_id + 1}/{NUM_SHADOW_MODELS}")

        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_all, y_all,
            test_size=TEST_SIZE,
            random_state=model_id,
            stratify=y_all
        )

        X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = TargetModel(input_dim=INPUT_DIM).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        if use_dp:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            model, optimizer, loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
            )

        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Extract features
        member_features = extract_attack_features(model, X_train, y_train)
        for f in member_features:
            attack_X.append(f)
            attack_y.append(1)

        nonmember_features = extract_attack_features(model, X_test, y_test)
        for f in nonmember_features:
            attack_X.append(f)
            attack_y.append(0)

        # Stats
        with torch.no_grad():
            train_acc = ((model(X_train).squeeze() > 0.5).float() == y_train).float().mean()
            test_acc = ((model(X_test).squeeze() > 0.5).float() == y_test).float().mean()
            member_losses = [f[2] for f in member_features]
            nonmember_losses = [f[2] for f in nonmember_features]

        eps_str = ""
        if use_dp:
            eps_str = f", ε: {privacy_engine.get_epsilon(delta=1e-5):.2f}"

        print(f"  Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}{eps_str}")
        print(f"  Avg loss — members: {np.mean(member_losses):.4f}, "
              f"non-members: {np.mean(nonmember_losses):.4f}")

    attack_X = np.array(attack_X)
    attack_y = np.array(attack_y)

    print(f"\nAttack dataset: {len(attack_X)} samples "
          f"({attack_y.sum():.0f} members, {len(attack_y) - attack_y.sum():.0f} non-members)")

    member_mask = attack_y == 1
    print(f"\nFeature means — Members:     {attack_X[member_mask].mean(axis=0).round(4)}")
    print(f"Feature means — Non-members: {attack_X[~member_mask].mean(axis=0).round(4)}")
    print(f"Feature stdev — Members:     {attack_X[member_mask].std(axis=0).round(4)}")
    print(f"Feature stdev — Non-members: {attack_X[~member_mask].std(axis=0).round(4)}")
    print(f"  (columns: prob0, prob1, loss, entropy)")

    os.makedirs("experiments", exist_ok=True)
    np.save(f"experiments/attack_features{suffix}.npy", attack_X)
    np.save(f"experiments/attack_labels{suffix}.npy", attack_y)

    print(f"\nAttack dataset saved to experiments/attack_features{suffix}.npy")


if __name__ == "__main__":
    main()
