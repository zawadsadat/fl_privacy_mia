"""
Shared configuration for the FL privacy project.
All scripts import from here so parameters stay consistent.

Change DATASET to switch between experiments.
"""

# --- Dataset selection ---
# Options: "breast_cancer", "heart_disease", "cardiotocography", "diabetes_hospital"
DATASET = "breast_cancer"


# --- Data split ---
# Per-dataset recommended splits:
#   breast_cancer:      0.95 (5% training, ~28 samples — stress test)
#   heart_disease:      0.90 (10% training, ~92 samples)
#   cardiotocography:   0.85 (15% training, ~319 samples)
#   diabetes_hospital:  0.85 (15% training, ~10,495 samples)
DATASET_SPLITS = {
    "breast_cancer": 0.95,
    "heart_disease": 0.90,
    "cardiotocography": 0.85,
    "diabetes_hospital": 0.85,
}
TEST_SIZE = DATASET_SPLITS.get(DATASET, 0.85)

# --- Model ---
INPUT_DIM = None  # set dynamically based on dataset

# --- Training ---
LR = 0.001
BATCH_SIZE = 16

# --- DP-SGD ---
NOISE_MULTIPLIER = 1.3
MAX_GRAD_NORM = 1.0
DP_DELTA = 1e-5

# --- Target model training ---
TARGET_EPOCHS = 50

# --- Shadow models ---
NUM_SHADOW_MODELS = 15
SHADOW_EPOCHS_NO_DP = 30
SHADOW_EPOCHS_DP = 50

# --- Attack model ---
ATTACK_NN_EPOCHS = 200
