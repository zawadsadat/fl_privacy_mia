import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier


# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

COLOR_NODP = "#d62728"
COLOR_DP = "#1f77b4"
COLOR_MEMBER = "#d62728"
COLOR_NONMEMBER = "#1f77b4"
COLOR_RANDOM = "#7f7f7f"
FEATURE_NAMES = ["prob0", "prob1", "loss", "entropy"]
FEATURE_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

os.makedirs("experiments/plots", exist_ok=True)


def load_and_prepare(suffix):
    X = np.load(f"experiments/attack_features{suffix}.npy")
    y = np.load(f"experiments/attack_labels{suffix}.npy")

    # Balance
    member_idx = np.where(y == 1)[0]
    nonmember_idx = np.where(y == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    m_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nm_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)

    balanced_idx = np.concatenate([m_sample, nm_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = X[balanced_idx], y[balanced_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, rf_probs)
    auc = roc_auc_score(y_test, rf_probs)
    acc = accuracy_score(y_test, (rf_probs > 0.5).astype(float))

    return {
        "X_raw": X, "y_raw": y,
        "X_test": X_test, "y_test": y_test,
        "rf": rf, "probs": rf_probs,
        "fpr": fpr, "tpr": tpr,
        "auc": auc, "acc": acc,
    }


# Load data
print("Loading experimental data...")
nodp = load_and_prepare("_nodp")
dp = load_and_prepare("_dp")


# Figure 1: ROC Curves
# ================================================================
print("Generating ROC curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Full ROC
ax1.plot(nodp["fpr"], nodp["tpr"], color=COLOR_NODP, lw=2.2,
         label=f'No DP (AUC = {nodp["auc"]:.3f})')
ax1.plot(dp["fpr"], dp["tpr"], color=COLOR_DP, lw=2.2,
         label=f'With DP (AUC = {dp["auc"]:.3f})')
ax1.plot([0, 1], [0, 1], color=COLOR_RANDOM, lw=1.2, ls="--",
         label="Random (AUC = 0.500)")
ax1.fill_between(nodp["fpr"], nodp["tpr"], alpha=0.08, color=COLOR_NODP)
ax1.fill_between(dp["fpr"], dp["tpr"], alpha=0.08, color=COLOR_DP)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve: MIA Attack Performance")
ax1.legend(loc="lower right")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_aspect("equal")

# Zoomed low-FPR region
ax2.plot(nodp["fpr"], nodp["tpr"], color=COLOR_NODP, lw=2.2,
         label=f"No DP", marker="o", markersize=3)
ax2.plot(dp["fpr"], dp["tpr"], color=COLOR_DP, lw=2.2,
         label=f"With DP", marker="o", markersize=3)
ax2.plot([0, 0.2], [0, 0.2], color=COLOR_RANDOM, lw=1.2, ls="--",
         label="Random")
ax2.axvline(x=0.05, color="#ff7f0e", ls=":", alpha=0.7, label="5% FPR")
ax2.axvline(x=0.10, color="#2ca02c", ls=":", alpha=0.7, label="10% FPR")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve: Low-FPR Region (Zoomed)")
ax2.legend(loc="upper left", fontsize=9)
ax2.set_xlim([0, 0.2])
ax2.set_ylim([0, 0.35])

plt.tight_layout()
plt.savefig("experiments/plots/01_roc_curves.png", dpi=200, bbox_inches="tight")
plt.close()



# Figure 2: Attack Performance Comparison Bar Chart
# ================================================================
print("Generating performance comparison...")

def get_tpr_at_fpr(fpr_arr, tpr_arr, target):
    idx = np.argmin(np.abs(fpr_arr - target))
    return tpr_arr[idx]

metrics = {
    "Accuracy": (nodp["acc"], dp["acc"]),
    "AUC": (nodp["auc"], dp["auc"]),
    "TPR@5%FPR": (
        get_tpr_at_fpr(nodp["fpr"], nodp["tpr"], 0.05),
        get_tpr_at_fpr(dp["fpr"], dp["tpr"], 0.05),
    ),
    "TPR@10%FPR": (
        get_tpr_at_fpr(nodp["fpr"], nodp["tpr"], 0.10),
        get_tpr_at_fpr(dp["fpr"], dp["tpr"], 0.10),
    ),
}

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(metrics))
width = 0.32

bars1 = ax.bar(x - width/2, [v[0] for v in metrics.values()],
               width, label="No DP", color=COLOR_NODP, alpha=0.85, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, [v[1] for v in metrics.values()],
               width, label="With DP", color=COLOR_DP, alpha=0.85, edgecolor="white", linewidth=0.5)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=COLOR_NODP)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=COLOR_DP)

ax.axhline(y=0.5, color=COLOR_RANDOM, ls="--", lw=1, alpha=0.6, label="Random baseline")
ax.set_xticks(x)
ax.set_xticklabels(metrics.keys())
ax.set_ylabel("Score")
ax.set_title("MIA Attack Metrics: DP vs No-DP")
ax.legend()
ax.set_ylim([0, max(max(v) for v in metrics.values()) + 0.08])

plt.tight_layout()
plt.savefig("experiments/plots/02_performance_comparison.png", dpi=200, bbox_inches="tight")
plt.close()


# Figure 3: Feature Distribution Box Plots
# ================================================================
print("Generating feature distributions...")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Attack Feature Distributions: Members vs Non-Members", fontsize=14, fontweight="bold", y=1.02)

for col, (suffix, label, data) in enumerate([(0, "Without DP", nodp), (1, "With DP", dp)]):
    X_raw = data["X_raw"]
    y_raw = data["y_raw"]
    member_mask = y_raw == 1

    for i, fname in enumerate(FEATURE_NAMES):
        ax = axes[col][i]

        member_vals = X_raw[member_mask, i]
        nonmember_vals = X_raw[~member_mask, i]

        bp = ax.boxplot(
            [member_vals, nonmember_vals],
            labels=["Member", "Non-member"],
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"markersize": 3, "alpha": 0.4},
        )
        bp["boxes"][0].set_facecolor(COLOR_MEMBER)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(COLOR_NONMEMBER)
        bp["boxes"][1].set_alpha(0.6)

        ax.set_title(f"{fname}", fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_ylabel(label, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("experiments/plots/03_feature_distributions.png", dpi=200, bbox_inches="tight")
plt.close()


# Figure 4: Feature Importance Comparison
# ================================================================
print("Generating feature importance comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# No DP
imp_nodp = nodp["rf"].feature_importances_
order_nodp = np.argsort(imp_nodp)
ax1.barh([FEATURE_NAMES[i] for i in order_nodp], imp_nodp[order_nodp],
         color=[FEATURE_COLORS[i] for i in order_nodp], alpha=0.85,
         edgecolor="white", linewidth=0.5)
for i, v in enumerate(imp_nodp[order_nodp]):
    ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10)
ax1.set_xlabel("Importance")
ax1.set_title("Without DP")
ax1.set_xlim([0, max(imp_nodp) + 0.06])

# With DP
imp_dp = dp["rf"].feature_importances_
order_dp = np.argsort(imp_dp)
ax2.barh([FEATURE_NAMES[i] for i in order_dp], imp_dp[order_dp],
         color=[FEATURE_COLORS[i] for i in order_dp], alpha=0.5,
         edgecolor="white", linewidth=0.5)
for i, v in enumerate(imp_dp[order_dp]):
    ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10)
ax2.set_xlabel("Importance")
ax2.set_title("With DP")
ax2.set_xlim([0, max(imp_nodp) + 0.06])  # same scale for comparison

fig.suptitle("Random Forest Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("experiments/plots/04_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()



# Figure 5: Loss Gap Visualization
# ================================================================
print("Generating loss gap visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

for ax, data, label, suffix in [
    (ax1, nodp, "Without DP", "_nodp"),
    (ax2, dp, "With DP", "_dp"),
]:
    X_raw = data["X_raw"]
    y_raw = data["y_raw"]
    member_mask = y_raw == 1

    member_losses = X_raw[member_mask, 2]
    nonmember_losses = X_raw[~member_mask, 2]

    # Histogram
    bins = np.linspace(0, min(2.0, max(nonmember_losses.max(), member_losses.max()) + 0.1), 50)
    ax.hist(member_losses, bins=bins, alpha=0.6, color=COLOR_MEMBER,
            label=f"Members (mean={member_losses.mean():.4f})", density=True)
    ax.hist(nonmember_losses, bins=bins, alpha=0.6, color=COLOR_NONMEMBER,
            label=f"Non-members (mean={nonmember_losses.mean():.4f})", density=True)
    ax.axvline(member_losses.mean(), color=COLOR_MEMBER, ls="--", lw=1.5)
    ax.axvline(nonmember_losses.mean(), color=COLOR_NONMEMBER, ls="--", lw=1.5)

    gap = abs(member_losses.mean() - nonmember_losses.mean())
    ax.set_title(f"{label}\nLoss gap = {gap:.4f}")
    ax.set_xlabel("Per-sample BCE Loss")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Loss Distribution: Members vs Non-Members", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("experiments/plots/05_loss_distributions.png", dpi=200, bbox_inches="tight")
plt.close()



# Figure 6: Combined Summary Figure
# ================================================================
print("Generating combined summary...")

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# (0,0) ROC Curves
ax = fig.add_subplot(gs[0, 0])
ax.plot(nodp["fpr"], nodp["tpr"], color=COLOR_NODP, lw=2,
        label=f'No DP (AUC={nodp["auc"]:.3f})')
ax.plot(dp["fpr"], dp["tpr"], color=COLOR_DP, lw=2,
        label=f'With DP (AUC={dp["auc"]:.3f})')
ax.plot([0, 1], [0, 1], color=COLOR_RANDOM, lw=1, ls="--")
ax.fill_between(nodp["fpr"], nodp["tpr"], alpha=0.06, color=COLOR_NODP)
ax.fill_between(dp["fpr"], dp["tpr"], alpha=0.06, color=COLOR_DP)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("(a) ROC Curves")
ax.legend(fontsize=8, loc="lower right")
ax.set_aspect("equal")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# (0,1) Performance bars
ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(metrics))
width = 0.32
ax.bar(x - width/2, [v[0] for v in metrics.values()], width,
       label="No DP", color=COLOR_NODP, alpha=0.85)
ax.bar(x + width/2, [v[1] for v in metrics.values()], width,
       label="With DP", color=COLOR_DP, alpha=0.85)
ax.axhline(y=0.5, color=COLOR_RANDOM, ls="--", lw=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(metrics.keys(), fontsize=8)
ax.set_title("(b) Attack Metrics")
ax.legend(fontsize=8)

# (0,2) Feature importance (no DP)
ax = fig.add_subplot(gs[0, 2])
order = np.argsort(imp_nodp)
ax.barh([FEATURE_NAMES[i] for i in order], imp_nodp[order],
        color=[FEATURE_COLORS[i] for i in order], alpha=0.85)
for i, v in enumerate(imp_nodp[order]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_title("(c) Feature Importance (No DP)")
ax.set_xlim([0, max(imp_nodp) + 0.08])

# (1,0) Loss distribution no-DP
ax = fig.add_subplot(gs[1, 0])
m_loss = nodp["X_raw"][nodp["y_raw"] == 1, 2]
nm_loss = nodp["X_raw"][nodp["y_raw"] == 0, 2]
bins = np.linspace(0, min(2.0, max(nm_loss.max(), m_loss.max()) + 0.1), 40)
ax.hist(m_loss, bins=bins, alpha=0.6, color=COLOR_MEMBER, label="Members", density=True)
ax.hist(nm_loss, bins=bins, alpha=0.6, color=COLOR_NONMEMBER, label="Non-members", density=True)
ax.axvline(m_loss.mean(), color=COLOR_MEMBER, ls="--", lw=1.5)
ax.axvline(nm_loss.mean(), color=COLOR_NONMEMBER, ls="--", lw=1.5)
ax.set_title(f"(d) Loss Distribution (No DP)\ngap = {abs(m_loss.mean()-nm_loss.mean()):.4f}")
ax.set_xlabel("BCE Loss")
ax.legend(fontsize=8)

# (1,1) Loss distribution with DP
ax = fig.add_subplot(gs[1, 1])
m_loss_dp = dp["X_raw"][dp["y_raw"] == 1, 2]
nm_loss_dp = dp["X_raw"][dp["y_raw"] == 0, 2]
bins_dp = np.linspace(0, min(2.0, max(nm_loss_dp.max(), m_loss_dp.max()) + 0.1), 40)
ax.hist(m_loss_dp, bins=bins_dp, alpha=0.6, color=COLOR_MEMBER, label="Members", density=True)
ax.hist(nm_loss_dp, bins=bins_dp, alpha=0.6, color=COLOR_NONMEMBER, label="Non-members", density=True)
ax.axvline(m_loss_dp.mean(), color=COLOR_MEMBER, ls="--", lw=1.5)
ax.axvline(nm_loss_dp.mean(), color=COLOR_NONMEMBER, ls="--", lw=1.5)
ax.set_title(f"(e) Loss Distribution (With DP)\ngap = {abs(m_loss_dp.mean()-nm_loss_dp.mean()):.4f}")
ax.set_xlabel("BCE Loss")
ax.legend(fontsize=8)

# (1,2) Summary table as text
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")

loss_gap_nodp = abs(m_loss.mean() - nm_loss.mean())
loss_gap_dp = abs(m_loss_dp.mean() - nm_loss_dp.mean())

table_data = [
    ["Metric", "No DP", "With DP"],
    ["AUC", f"{nodp['auc']:.4f}", f"{dp['auc']:.4f}"],
    ["Accuracy", f"{nodp['acc']:.4f}", f"{dp['acc']:.4f}"],
    ["TPR@5%", f"{metrics['TPR@5%FPR'][0]:.4f}", f"{metrics['TPR@5%FPR'][1]:.4f}"],
    ["TPR@10%", f"{metrics['TPR@10%FPR'][0]:.4f}", f"{metrics['TPR@10%FPR'][1]:.4f}"],
    ["Loss Gap", f"{loss_gap_nodp:.4f}", f"{loss_gap_dp:.4f}"],
]

table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header
for j in range(3):
    table[0, j].set_facecolor("#2B5797")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[i, j].set_facecolor("#f0f4f8")

ax.set_title("(f) Results Summary", pad=20)

fig.suptitle("Membership Inference Attack: Effect of Differential Privacy",
             fontsize=15, fontweight="bold", y=1.01)

plt.savefig("experiments/plots/06_combined_summary.png", dpi=200, bbox_inches="tight")
plt.close()


print(f"\nAll plots saved to experiments/plots/")
print("  01_roc_curves.png")
print("  02_performance_comparison.png")
print("  03_feature_distributions.png")
print("  04_feature_importance.png")
print("  05_loss_distributions.png")
print("  06_combined_summary.png")
