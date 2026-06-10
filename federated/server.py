"""
server.py -- AgriFL Federated Learning
========================================
Server-side strategy configuration.

Uses FedAvg (Federated Averaging) as the aggregation strategy.
FedAvg aggregates client weights proportionally to their dataset size:

    w_global = sum(n_k / n_total * w_k)   for each client k

Custom callbacks log per-round accuracy and loss for reporting.
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate evaluate() metrics from all clients using weighted average.

    Flower passes a list of (num_examples, metrics_dict) tuples.
    We compute accuracy weighted by number of examples per client.
    """
    accuracies  = [m["accuracy"] * n for n, m in metrics]
    total       = sum(n for n, _ in metrics)
    return {"accuracy": sum(accuracies) / total}


def build_strategy(
    num_clients: int,
    min_fit_clients: int = None,
    min_eval_clients: int = None,
    fraction_fit: float = 1.0,
):
    """
    Build and return a FedAvg strategy.

    Parameters
    ----------
    num_clients      : total number of FL clients (soil partitions)
    min_fit_clients  : minimum clients required for training round
    fraction_fit     : fraction of clients sampled each round (1.0 = all)
    """
    min_fit_clients  = min_fit_clients  or num_clients
    min_eval_clients = min_eval_clients or num_clients

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return strategy


def get_server_config(num_rounds: int) -> fl.server.ServerConfig:
    """Return the server configuration with a given number of FL rounds."""
    return fl.server.ServerConfig(num_rounds=num_rounds)
