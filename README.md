# Privacy Leakage in Federated Learning: A Membership Inference Attack Analysis with Differential Privacy Defenses

> **Course:** CISC6880 — Fordham University  
> **Authors:** Muhammad Zawad Mahmud · Samiha Islam  

---

## Overview

This repository contains the full implementation for our empirical study of **membership inference attacks (MIA)** against federated learning models in a medical classification setting. We evaluate whether an honest-but-curious federated client can determine if a specific patient record was used in training, and measure how effectively **DP-SGD** (Differential Privacy Stochastic Gradient Descent) defends against such attacks.

### Key Results

| Condition | Best Attack AUC | Loss Gap | Model Accuracy |
|-----------|----------------|----------|---------------|
| No DP (centralized) | 0.6030 | 0.2939 | >95% |
| With DP-SGD (ε≈26.51) | 0.4979 | 0.0045 | 66.36% |
| FL global model (FedAvg) | 0.5407 | — | 95.01% |

**DP-SGD reduces the member/non-member loss gap by 65.3×**, effectively neutralizing shadow-model-based MIAs at the cost of ~29 percentage points of accuracy.

---

## Repository Structure

```
fl_privacy/
│
├── models/
│   ├── model.py                  # Shared model architecture (TargetModel + AttackModel)
│   └── train_target_model.py     # Train centralized target with DP-SGD (Opacus)
│
├── attacks/
│   ├── shadow_models.py          # Train 15 shadow models, extract attack features
│   ├── train_attack_model.py     # Train 3 attack classifiers (RF, NN, threshold)
│   ├── evaluate_mia.py           # Evaluate attack against any saved model
│   ├── compare_dp_effect.py      # Print DP vs no-DP comparison table
│   └── generate_plots.py         # Generate publication-ready figures
│
├── federated/
│   ├── server.py                 # FedAvg aggregation server (Flower)
│   └── client.py                 # FL client with local training
│
├── utils/
│   ├── config.py                 # Central config (hyperparameters, DP settings)
│   └── data_loader.py            # Breast Cancer Wisconsin dataset loader
│
├── data/
│
├── experiments/                  # Auto-created — stores saved models and results
│   ├── target_model.pt
│   ├── fl_global_model.pt
│   ├── attack_features_nodp.npy
│   ├── attack_features_dp.npy
│   └── plots/
│
├── requirements.txt
└── README.md
```

---

## Threat Model

We define three adversary tiers:

| Adversary | Access | Goal |
|-----------|--------|------|
| **Adv. 1** — External observer | Blockchain ledger metadata only | Infer participation patterns |
| **Adv. 2** — Honest-but-curious client ★ | Global model after each round | Cross-silo membership inference |
| **Adv. 3** — Malicious coordinator | All client updates | Gradient inversion |

★ **Adversary 2 is evaluated empirically in this work.**

---

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA (recommended) or CPU
- Tested on: Ubuntu 22.04, PyTorch 2.5.1+cu121

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fl_privacy.git
cd fl_privacy

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
opacus>=1.4.0
flwr>=1.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## Running the Full Pipeline

### Step 1 — Train the target model (centralized, with DP-SGD)

```bash
python models/train_target_model.py
```

**Expected output:**
```
Training samples: 28, Test samples: 541
Epoch 0 Loss: 0.6886
...
Test Accuracy: 0.6636
Privacy budget: ε = 26.51, δ = 1e-05
Target model saved.
```

---

### Step 2 — Train shadow models and extract attack features

**Without DP (baseline):**
```bash
python attacks/shadow_models.py --no-dp
```

**With DP-SGD (mirrors target):**
```bash
python attacks/shadow_models.py --dp
```

Each run trains 15 shadow models and saves attack features to `experiments/`.  
Watch the **member vs non-member loss gap** in the output — a large gap means the attack has signal.

---

### Step 3 — Train attack classifiers

**Without DP:**
```bash
python attacks/train_attack_model.py --no-dp
```

**With DP:**
```bash
python attacks/train_attack_model.py --dp
```

Reports three attacks side by side:
- **Attack 1:** Loss threshold baseline
- **Attack 2:** Random Forest (200 trees, max depth 5)
- **Attack 3:** Neural network (4→64→32→16→1, 200 epochs)

---

### Step 4 — Compare DP vs no-DP

```bash
python attacks/compare_dp_effect.py
```

Prints a summary table showing AUC, accuracy, TPR@FPR, and loss gap reduction factor.

---

### Step 5 — Run the federated simulation

Open **4 terminals** in the project directory:

**Terminal 1 — Start the aggregation server:**
```bash
python federated/server.py
```

**Terminals 2, 3, 4 — Start each hospital client:**
```bash
python federated/client.py --client-id 0
python federated/client.py --client-id 1
python federated/client.py --client-id 2
```

Wait for all 10 rounds to complete. The server saves `experiments/fl_global_model.pt` after round 10 and prints a round-by-round accuracy history.

**Expected final accuracy:** ~95.01% after 10 rounds.

---

### Step 6 — Evaluate MIA against the FL global model

```bash
# Using the no-DP attack classifier
python attacks/evaluate_mia.py \
    --model experiments/fl_global_model.pt \
    --attack-model experiments/attack_model_nodp.pt

# Using the DP attack classifier
python attacks/evaluate_mia.py \
    --model experiments/fl_global_model.pt \
    --attack-model experiments/attack_model_dp.pt
```

---

### Step 7 — Generate plots

```bash
python attacks/generate_plots.py
```

Saves 6 publication-ready figures to `experiments/plots/`:

| File | Content |
|------|---------|
| `01_roc_curves.png` | ROC curves — DP vs no-DP overlay |
| `02_performance_comparison.png` | Bar chart — all metrics, all attacks |
| `04_feature_importance.png` | RF feature importances — DP vs no-DP |
| `05_loss_distributions.png` | Member vs non-member loss histograms |

---

## Configuration

All hyperparameters are centralized in `utils/config.py`:

```python
# Data
TEST_SIZE        = 0.95      # 5% training split (~28 samples)

# Model architecture
HIDDEN_DIMS      = [256, 128, 64, 32]

# DP-SGD (Opacus)
NOISE_MULTIPLIER = 1.3
MAX_GRAD_NORM    = 1.0
DELTA            = 1e-5
EPOCHS           = 50
BATCH_SIZE       = 16

# Shadow models
NUM_SHADOW       = 15

# Federated learning (Flower)
NUM_CLIENTS      = 3
NUM_ROUNDS       = 10
LOCAL_EPOCHS     = 5
```

---

## Results Summary

### Centralized model — DP vs no-DP MIA comparison

| Metric | No DP | With DP | Δ |
|--------|-------|---------|---|
| Attack AUC (RF) | 0.6030 | 0.4979 | −0.1051 |
| Attack Accuracy | 0.5873 | 0.4960 | −0.0913 |
| TPR @ 5% FPR | 0.0476 | 0.0397 | −0.0079 |
| TPR @ 10% FPR | 0.1190 | 0.0873 | −0.0317 |
| Loss gap | 0.2939 | 0.0045 | −0.2894 |
| Reduction factor | — | — | **65.3×** |
| Model accuracy | >95% | 66.36% | ~−29pp |

### Centralized vs federated MIA comparison

| Target model | Attacker | AUC | Acc. | TPR@10% |
|-------------|---------|-----|------|---------|
| Centralized (no DP) | RF no-DP | 0.6030 | 0.5873 | 0.1190 |
| Centralized (DP) | RF DP | 0.4979 | 0.4960 | 0.0873 |
| FL global model | no-DP atk | 0.5407 | 0.1705 | 0.0000 |
| FL global model | DP atk | 0.5145 | 0.0967 | 0.1071 |

---

<!-- ## Citation

If you use this code, please cite our paper:

```bibtex
@article{yourname2025fl_privacy,
  title   = {Privacy Leakage in Federated Learning: A Membership Inference
             Attack Analysis with Differential Privacy Defenses},
  author  = {Author 1 and Author 2},
  journal = {CISC6880 Course Project, Fordham University},
  year    = {2025}
}
```

--- -->

## References

- McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks from Decentralized Data
- Shokri et al. (2017) — Membership Inference Attacks Against Machine Learning Models
- Abadi et al. (2016) — Deep Learning with Differential Privacy
- Nasr et al. (2019) — Comprehensive Privacy Analysis of Deep Learning
- Yousefpour et al. (2021) — Opacus: User-Friendly Differential Privacy Library in PyTorch

---

## License

MIT License. See `LICENSE` for details.
