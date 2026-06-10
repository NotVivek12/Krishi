"""
aggregation.py -- AgriFL Federated Learning
=============================================
Server-side aggregation strategies.

Currently implements:
  - FedAvg  (McMahan et al., 2017)  -- weighted average of client weights

Usage
-----
    from federated.aggregation import fedavg

    new_global_weights = fedavg(client_weights, client_sizes)
"""

import numpy as np


def fedavg(client_weights: list, client_sizes: list) -> list:
    """
    Federated Averaging (McMahan et al., 2017).

    Aggregates client model weights proportionally to their dataset size:

        w_global = Σ_k  (n_k / n_total) * w_k

    Parameters
    ----------
    client_weights : list of list[np.ndarray]
        One entry per participating client; each entry is a list of weight
        arrays (one array per model layer), as returned by model.get_weights().

    client_sizes : list of int
        Number of training samples for each client.  Must be the same length
        as `client_weights`.

    Returns
    -------
    list[np.ndarray]
        Aggregated weight arrays with the same structure as a single client's
        weight list.

    Raises
    ------
    ValueError
        If `client_weights` and `client_sizes` have different lengths, or if
        fewer than one client is provided.
    """
    if len(client_weights) != len(client_sizes):
        raise ValueError(
            f"client_weights ({len(client_weights)}) and "
            f"client_sizes ({len(client_sizes)}) must have the same length."
        )
    if len(client_weights) == 0:
        raise ValueError("At least one client must participate in aggregation.")

    total = sum(client_sizes)
    if total == 0:
        raise ValueError("Total dataset size across all clients must be > 0.")

    n_layers = len(client_weights[0])
    avg_weights = []

    for layer_idx in range(n_layers):
        layer_avg = np.sum(
            [
                (client_sizes[k] / total) * np.array(client_weights[k][layer_idx])
                for k in range(len(client_weights))
            ],
            axis=0,
        )
        avg_weights.append(layer_avg)

    return avg_weights


def fedavg_simple(client_weights: list) -> list:
    """
    Unweighted (simple) Federated Averaging — treats all clients equally.

    Equivalent to fedavg with uniform client_sizes.  Useful for quick
    sanity checks where dataset sizes are roughly equal.

    Parameters
    ----------
    client_weights : list of list[np.ndarray]

    Returns
    -------
    list[np.ndarray]
    """
    n_clients = len(client_weights)
    uniform_sizes = [1] * n_clients
    return fedavg(client_weights, uniform_sizes)
