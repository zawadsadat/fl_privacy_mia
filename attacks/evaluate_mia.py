import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from models.model import TargetModel, AttackModel
from utils.data_loader import load_data


def extract_features_for_sample(model, x, y_true):
    
    with torch.no_grad():
        p = model(x.unsqueeze(0)).squeeze()
        confidence = p.item()
        prob0 = 1 - confidence
        prob1 = confidence
        loss = F.binary_cross_entropy(p, y_true, reduction='none').item()
        entropy = -(
            confidence * math.log(confidence + 1e-10) +
            (1 - confidence) * math.log(1 - confidence + 1e-10)
        )
    return [prob0, prob1, loss, entropy]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to target model state dict")
    parser.add_argument("--attack-model", type=str,
                        default="experiments/attack_model.pt",
                        help="Path to trained attack model")
    args = parser.parse_args()

    # Load target model
    target = TargetModel(input_dim=30)
    target.load_state_dict(torch.load(args.model, map_location="cpu"))
    target.eval()

    # Load attack model
    attacker = AttackModel(input_dim=4)
    attacker.load_state_dict(torch.load(args.attack_model, map_location="cpu"))
    attacker.eval()

    # Load data — same split as training
    X_train, X_test, y_train, y_test = load_data()

    # Build evaluation set:
    # Members = training samples, Non-members = test samples
    all_features = []
    all_labels = []  # 1 = member, 0 = non-member

    print("Extracting features from training set (members)...")
    for i in range(len(X_train)):
        features = extract_features_for_sample(target, X_train[i], y_train[i])
        all_features.append(features)
        all_labels.append(1)

    print("Extracting features from test set (non-members)...")
    for i in range(len(X_test)):
        features = extract_features_for_sample(target, X_test[i], y_test[i])
        all_features.append(features)
        all_labels.append(0)

    X_attack = torch.tensor(all_features, dtype=torch.float32)
    y_attack = np.array(all_labels)

    # Run attack
    with torch.no_grad():
        preds = attacker(X_attack).squeeze().numpy()

    predicted = (preds > 0.5).astype(float)

    acc = accuracy_score(y_attack, predicted)
    auc = roc_auc_score(y_attack, preds)

    print(f"\n{'=' * 40}")
    print(f"MIA Evaluation: {args.model}")
    print(f"{'=' * 40}")
    print(f"Members:     {(y_attack == 1).sum()}")
    print(f"Non-members: {(y_attack == 0).sum()}")
    print(f"Attack Accuracy: {acc:.4f}")
    print(f"Attack AUC:      {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_attack, preds)
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        print(f"TPR @ {target_fpr*100:.0f}% FPR: {tpr[idx]:.4f}")

    # Baseline comparison
    print(f"\nBaseline (random guess): {(y_attack == 1).mean():.4f}")
    print(f"Attack advantage:        {acc - 0.5:.4f}")


if __name__ == "__main__":
    main()
