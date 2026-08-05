"""
Flower FL client.

Each client gets a disjoint partition of the training data.

Usage:
    python federated/client.py --client-id 0
    python federated/client.py --client-id 1
    python federated/client.py --client-id 2
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.model import TargetModel
from utils.data_loader import load_data, get_input_dim
from utils.config import LR


NUM_CLIENTS = 3
LOCAL_EPOCHS = 5
INPUT_DIM = get_input_dim()


def partition_data(X_train, y_train, num_clients, client_id):
    """Split training data into disjoint partitions."""
    n = len(X_train)
    indices = list(range(n))
    np.random.RandomState(42).shuffle(indices)

    chunk_size = n // num_clients
    start = client_id * chunk_size
    end = start + chunk_size if client_id < num_clients - 1 else n

    client_indices = indices[start:end]
    return X_train[client_indices], y_train[client_indices]


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id):
        self.client_id = client_id
        self.model = TargetModel(input_dim=INPUT_DIM)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        X_train, X_test, y_train, y_test = load_data()

        self.X_train, self.y_train = partition_data(
            X_train, y_train, NUM_CLIENTS, client_id
        )
        self.X_test = X_test
        self.y_test = y_test

        print(f"Client {client_id}: {len(self.X_train)} training samples, "
              f"{len(self.X_test)} test samples, input_dim={INPUT_DIM}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        for epoch in range(LOCAL_EPOCHS):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train).squeeze()
            loss = self.criterion(outputs, self.y_train)
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        with torch.no_grad():
            preds = self.model(self.X_test).squeeze()
            loss = self.criterion(preds, self.y_test)
            predicted = (preds > 0.5).float()
            accuracy = (predicted == self.y_test).sum().item() / len(self.y_test)

        return float(loss), len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    args = parser.parse_args()

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=FlowerClient(args.client_id)
    )
