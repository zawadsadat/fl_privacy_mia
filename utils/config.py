# Data split
# How much data is held out as "non-member" test set.
# Higher = smaller training set = more memorization = stronger MIA signal.
# 0.95 means only 5% (~28 samples) used for training.
TEST_SIZE = 0.95

# Model
INPUT_DIM = 30          # Breast Cancer features

# Training
LR = 0.001
BATCH_SIZE = 16         # smaller batches for tiny training set

# DP-SGD
NOISE_MULTIPLIER = 1.3
MAX_GRAD_NORM = 1.0
DP_DELTA = 1e-5

# Target model training
TARGET_EPOCHS = 50

# Shadow models
NUM_SHADOW_MODELS = 15
SHADOW_EPOCHS_NO_DP = 30
SHADOW_EPOCHS_DP = 50

# Attack model
ATTACK_NN_EPOCHS = 200
