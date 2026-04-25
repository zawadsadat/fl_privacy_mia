import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

from models.model import TargetModel
from utils.data_loader import load_data
from utils.config import (
    TARGET_EPOCHS as EPOCHS, BATCH_SIZE, LR,
    NOISE_MULTIPLIER, MAX_GRAD_NORM, DP_DELTA, INPUT_DIM,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data
X_train, X_test, y_train, y_test = load_data()
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model + DP
model = TargetModel(input_dim=INPUT_DIM).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
)


# Training
for epoch in range(EPOCHS):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")


# Evaluation
with torch.no_grad():
    predictions = model(X_test).squeeze()
    predicted = (predictions > 0.5).float()
    accuracy = (predicted == y_test).sum().item() / len(y_test)

print(f"Test Accuracy: {accuracy:.4f}")

epsilon = privacy_engine.get_epsilon(delta=DP_DELTA)
print(f"Privacy budget: ε = {epsilon:.2f}, δ = {DP_DELTA}")


# Save
os.makedirs("experiments", exist_ok=True)
torch.save(model.state_dict(), "experiments/target_model.pt")
torch.save(predictions, "experiments/predictions.pt")

print("Target model saved.")
