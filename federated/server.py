import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flwr as fl
import torch
import numpy as np

from models.model import TargetModel


NUM_ROUNDS = 10
MIN_CLIENTS = 3


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None and server_round == NUM_ROUNDS:
            print(f"\nRound {server_round}: Saving final global model...")

            parameters, _ = aggregated
            ndarrays = fl.common.parameters_to_ndarrays(parameters)

            # Reconstruct state dict
            model = TargetModel(input_dim=30)
            state_dict = {}
            for key, ndarray in zip(model.state_dict().keys(), ndarrays):
                state_dict[key] = torch.tensor(ndarray)
            model.load_state_dict(state_dict)

            os.makedirs("experiments", exist_ok=True)
            torch.save(model.state_dict(), "experiments/fl_global_model.pt")
            print("Global model saved to experiments/fl_global_model.pt")

        return aggregated


strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="127.0.0.1:8081",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
