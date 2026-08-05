# Privacy Auditing Framework for Federated Medical Learning

> Membership Inference Attacks · DP-SGD · FMPAS Privacy Audit Score

Empirical privacy-auditing framework that evaluates membership inference risk in federated learning across **4 medical datasets**, **12 attack strategies** (7 black-box + 5 white-box), non-IID data heterogeneity, and differential privacy configurations. Introduces the **Federated Membership Privacy Audit Score (FMPAS)** — a single-number risk metric with deployment recommendations.

---

## Key Results

| Dataset | Samples | Train | No-DP AUC | DP AUC | Loss Gap Red. | DP Acc | ε |
|---|---|---|---|---|---|---|---|
| Breast Cancer | 569 | 28 | 0.634 | 0.493 | 55× | 62.7% | 21.22 |
| Heart Disease (4 centers) | 920 | 92 | 0.687 | 0.518 | 34× | 62.4% | 14.21 |
| Cardiotocography | 2,126 | 318 | 0.547 | 0.538 | ~0 | 94.8% | 6.92 |
| **Diabetes Hospital** | **69,973** | **10,495** | **0.611** | **0.500** | **111×** | **91.0%** | **0.94** |

**On Diabetes, DP-SGD eliminates all measurable leakage while improving accuracy by 3.7%.**

### FMPAS Deployment Recommendations

| Dataset | No-DP Risk | σ* for Risk ≤ 0.01 | Accuracy at σ* |
|---|---|---|---|
| Diabetes | 0.163 | 0.5 | 91.0% |
| Cardiotocography | 0.077 | 0.8 | 99.8% |
| Breast Cancer | 0.238 | 0.5† | 88.9%† |
| Heart Disease | 0.310 | — | — |

† Single-seed estimate. — = noisy, needs multi-seed.

---

## Repository Structure

```
fl_privacy/
├── models/
│   ├── model.py                    # TargetModel (d_in→256→128→64→32→1) + AttackModel
│   └── train_target_model.py       # Train with/without DP-SGD (Opacus)
│
├── attacks/
│   ├── shadow_models.py            # Train 15 shadow models, extract features
│   ├── train_attack_model.py       # Train attack classifier on shadow features
│   ├── run_all_attacks.py          # 7 black-box attacks
│   ├── whitebox_attack.py          # 5 white-box gradient attacks
│   ├── evaluate_mia.py             # Attack evaluation / summary reporting
│   ├── compute_all_metrics.py      # Full metrics: AUC, advantage, PPV, TPR@FPR
│   ├── compare_dp_effect.py        # No-DP vs DP side-by-side comparison
│   └── generate_plots.py           # plots per dataset
│
├── experiments/
│   ├── run_epsilon_sweep.py        # Privacy budget sweep (8 noise levels)
│   ├── plot_epsilon_sweep.py       # Epsilon sweep figures
│   ├── run_noniid_sweep.py         # Dirichlet non-IID sweep (α × K configs)
│   ├── plot_noniid_results.py      # Non-IID sweep figures
│   ├── plot_whitebox.py            # White-box comparison plots
│   └── compute_audit_score.py      # FMPAS Risk score + recommendations
│
├── federated/
│   ├── server.py                   # FedAvg server (Flower, 10 rounds)
│   └── client.py                   # FL client (--client-id 0/1/2)
│
├── utils/
│   ├── config.py                   # Dataset selector + all hyperparameters
│   ├── data_loader.py              # Router: loads correct dataset
│   ├── data_loader_heart_disease.py
│   ├── data_loader_cardiotocography.py
│   ├── data_loader_diabetes_hospital.py
│   └── partition.py                # Dirichlet non-IID client partitioning
│
├── requirements.txt
└── README.md
```

---

## Threat Model

| Adversary | Access | Attack Type | # Attacks |
|---|---|---|---|
| Honest-but-curious client | Global model predictions | Black-box | 7 |
| Malicious coordinator | Client gradients | White-box | 5 |

---

## 12 Attack Strategies

**Black-box (Adversary 1):**
1. Loss threshold
2. Confidence threshold
3. Entropy threshold
4. Calibrated loss
5. LiRA (likelihood ratio)
6. Random Forest
7. Neural Network

**White-box (Adversary 2):**
8. Gradient norm (L2)
9. Gradient norm (L1)
10. Loss threshold (WB baseline)
11. RF on gradient features
12. RF on all features (BB + WB)

---

## Setup

```bash
# Clone
git clone https://github.com/zawadsadat/fl_privacy_mia.git
cd fl_privacy_mia

# Environment
conda create -n fl_privacy python=3.10 -y
conda activate fl_privacy
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.5.1+ (CUDA 12.1)
- Opacus, scikit-learn, Flower, matplotlib, pandas

---

## Running the Pipeline

### 1. Select dataset
Edit `utils/config.py`:
```python
DATASET = "diabetes_hospital"  # or breast_cancer, heart_disease, cardiotocography
```

### 2. Train and evaluate
```bash
# Target models
python models/train_target_model.py --no-dp
python models/train_target_model.py --dp

# Shadow models
python attacks/shadow_models.py --no-dp
python attacks/shadow_models.py --dp

# Attack classifier (trained on shadow features)
python attacks/train_attack_model.py --no-dp
python attacks/train_attack_model.py --dp

# 7 black-box attacks
python attacks/run_all_attacks.py --no-dp
python attacks/run_all_attacks.py --dp

# 5 white-box attacks
python attacks/whitebox_attack.py --no-dp
python attacks/whitebox_attack.py --dp

# Attack evaluation summary
python attacks/evaluate_mia.py --no-dp
python attacks/evaluate_mia.py --dp

# Comprehensive metrics + CSV
python attacks/compute_all_metrics.py --no-dp
python attacks/compute_all_metrics.py --dp

# No-DP vs DP comparison
python attacks/compare_dp_effect.py

# Plots
python attacks/generate_plots.py
python experiments/plot_whitebox.py
```

### 3. Privacy budget sweep
```bash
python experiments/run_epsilon_sweep.py
python experiments/plot_epsilon_sweep.py
```

### 4. Non-IID sweep
```bash
python experiments/run_noniid_sweep.py --alphas 0.1 0.5 1.0 10.0 --clients 3 5 10 20
python experiments/plot_noniid_results.py
```

### 5. FL simulation (4 terminals)
```bash
# Terminal 1
python federated/server.py
# Terminals 2-4
python federated/client.py --client-id 0
python federated/client.py --client-id 1
python federated/client.py --client-id 2
```

### 6. FMPAS audit score
```bash
python experiments/compute_audit_score.py
```

---

## Configuration

All parameters in `utils/config.py`:

```python
DATASET = "diabetes_hospital"
DATASET_SPLITS = {
    "breast_cancer": 0.95,
    "heart_disease": 0.90,
    "cardiotocography": 0.85,
    "diabetes_hospital": 0.85,
}
NOISE_MULTIPLIER = 1.3
MAX_GRAD_NORM = 1.0
DP_DELTA = 1e-5
TARGET_EPOCHS = 50
NUM_SHADOW_MODELS = 15
```

---

## References

- Shokri et al. (2017) — Membership Inference Attacks Against Machine Learning Models
- Carlini et al. (2022) — Membership Inference Attacks From First Principles
- Yeom et al. (2018) — Privacy Risk in Machine Learning
- Abadi et al. (2016) — Deep Learning with Differential Privacy
- McMahan et al. (2017) — Communication-Efficient Learning (FedAvg)
- Jayaraman & Evans (2019) — Evaluating Differentially Private ML in Practice

---

## License

Apache-2.0
