"""
straggler.py -- AgriFL Federated Learning
==========================================
Straggler detection and filtering for the FL simulation.

A "straggler" is a client whose local training takes longer than a
configurable timeout threshold.  In a real distributed system this happens
due to network latency, hardware differences, or data skew.  Here we
simulate it by:

  1. Measuring wall-clock time around each client's model.fit() call.
  2. Optionally injecting artificial slowness (via time.sleep) to test
     the straggler path without needing slow hardware.
  3. Skipping any client whose training time exceeds the timeout, so the
     server can still aggregate from the remaining fast clients.

Usage
-----
    from ml.straggler import StragglerSimulator

    sim = StragglerSimulator(timeout_seconds=30.0, slow_client_prob=0.0)
    result = sim.train_client(client, local_model, local_epochs, batch_size)
    if result is not None:
        weights, n_samples = result
"""

import time
import random
import numpy as np


class StragglerSimulator:
    """
    Wraps the local-training step with timeout-based straggler detection.

    Parameters
    ----------
    timeout_seconds  : float
        Maximum wall-clock seconds allowed for a client's training.
        Clients that exceed this are classified as stragglers and dropped.
    slow_client_prob : float  [0.0 – 1.0]
        Probability that a given client will be artificially slowed down
        (simulates real-world heterogeneity).  Set to 0.0 in production runs.
    slow_multiplier  : float
        How many extra seconds to sleep when a client is simulated as slow.
        E.g. 3.0 means the client sleeps for 3× the actual training time.
    seed             : int | None
        Random seed for reproducible straggler injection.
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        slow_client_prob: float = 0.2,
        slow_multiplier: float = 3.0,
        seed: int = 42,
    ):
        self.timeout_seconds  = timeout_seconds
        self.slow_client_prob = slow_client_prob
        self.slow_multiplier  = slow_multiplier
        self._rng             = random.Random(seed)

    def train_client(
        self,
        client: dict,
        local_model,
        local_epochs: int,
        batch_size: int,
        verbose: int = 0,
    ):
        """
        Train `local_model` on `client`'s data, respecting the timeout.

        Parameters
        ----------
        client       : dict with keys X_train, y_train, n_train, id, soil
        local_model  : compiled Keras model (weights already set to global)
        local_epochs : int
        batch_size   : int
        verbose      : int (Keras verbosity)

        Returns
        -------
        (weights, n_samples) : tuple
            weights   -- list[np.ndarray] updated model weights
            n_samples -- int number of training samples used
        None
            If the client is a straggler (timed out).
        """
        # Decide if this client will be artificially slowed
        is_slow = self._rng.random() < self.slow_client_prob

        t_start = time.time()

        local_model.fit(
            client['X_train'],
            client['y_train'],
            epochs=local_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        t_train = time.time() - t_start

        if is_slow:
            sleep_time = t_train * self.slow_multiplier
            time.sleep(sleep_time)

        t_total = time.time() - t_start

        if t_total > self.timeout_seconds:
            print(
                f"  [STRAGGLER] Client {client['id']} [{client['soil'][:28]}]"
                f" timed out ({t_total:.1f}s > {self.timeout_seconds}s) — skipped."
            )
            return None  # signal: drop this client from aggregation

        return local_model.get_weights(), client['n_train']


def filter_stragglers(results: list) -> tuple:
    """
    Separate completed client results from straggler None values.

    Parameters
    ----------
    results : list of (weights, n_samples) | None
        Raw output list from calling StragglerSimulator.train_client()
        for each client.

    Returns
    -------
    (client_weights, client_sizes, n_stragglers) : tuple
        client_weights -- list of weight lists from responding clients
        client_sizes   -- corresponding list of dataset sizes
        n_stragglers   -- number of clients that were dropped
    """
    client_weights = []
    client_sizes   = []
    n_stragglers   = 0

    for result in results:
        if result is None:
            n_stragglers += 1
        else:
            weights, n = result
            client_weights.append(weights)
            client_sizes.append(n)

    return client_weights, client_sizes, n_stragglers


def straggler_summary(n_total: int, n_stragglers: int) -> str:
    """Return a human-readable straggler summary string."""
    n_completed = n_total - n_stragglers
    pct = 100.0 * n_stragglers / n_total if n_total > 0 else 0.0
    return (
        f"Straggler report: {n_stragglers}/{n_total} clients dropped "
        f"({pct:.1f}%) — aggregating from {n_completed} clients."
    )
