"""
client.py -- AgriFL Federated Learning
========================================
Flower client implementation.

Each FlowerClient wraps a Keras DNN and represents a single farm region
(identified by its dominant soil type). During federation:
  - fit()      : trains locally on the client's soil partition
  - evaluate() : evaluates the global model on local data
"""

import numpy as np
import flwr as fl
from sklearn.model_selection import train_test_split

from federated.utils import build_dnn, get_model_weights, set_model_weights


class AgriFlClient(fl.client.NumPyClient):
    """
    A Flower NumPyClient representing one farm region (soil partition).

    Parameters
    ----------
    client_id   : int   -- numeric ID (1-N)
    soil_name   : str   -- soil type this client represents
    X           : ndarray (n_samples, n_features)
    y           : ndarray (n_samples,)
    num_classes : int
    local_epochs: int   -- number of local training epochs per round
    """

    def __init__(
        self,
        client_id: int,
        soil_name: str,
        X: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        local_epochs: int = 3,
    ):
        self.client_id   = client_id
        self.soil_name   = soil_name
        self.num_classes = num_classes
        self.local_epochs = local_epochs

        # Local train/val split (80/20)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
            if len(np.unique(y)) > 1 else None
        )

        # Build local model (weights will be overwritten by server each round)
        self.model = build_dnn(
            input_dim=X.shape[1],
            num_classes=num_classes,
        )

    # ── Flower interface ─────────────────────────────────────────────────────

    def get_parameters(self, config):
        """Return current local model weights to the server."""
        return get_model_weights(self.model)

    def fit(self, parameters, config):
        """
        Receive global weights from server, train locally, return updated weights.

        Called every federated round.
        """
        # 1. Apply global model weights
        set_model_weights(self.model, parameters)

        # 2. Local training
        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.local_epochs,
            batch_size=64,
            validation_data=(self.X_val, self.y_val),
            verbose=0,
        )

        # 3. Return updated weights + metadata
        updated_weights = get_model_weights(self.model)
        n_samples = len(self.X_train)
        metrics   = {"soil": self.soil_name, "client_id": self.client_id}

        return updated_weights, n_samples, metrics

    def evaluate(self, parameters, config):
        """
        Evaluate the global model on this client's local validation data.
        Returns (loss, n_samples, metrics_dict).
        """
        set_model_weights(self.model, parameters)

        loss, accuracy = self.model.evaluate(
            self.X_val, self.y_val, verbose=0
        )

        print(f"  [Client {self.client_id} | {self.soil_name}] "
              f"val_loss={loss:.4f}  val_acc={accuracy:.4f}")

        return float(loss), len(self.X_val), {"accuracy": float(accuracy)}


def make_client_fn(partitions: dict, num_classes: int, local_epochs: int = 3):
    """
    Factory that returns a client_fn compatible with flwr.simulation.

    Parameters
    ----------
    partitions   : dict  { soil_name: (X, y) }
    num_classes  : int
    local_epochs : int

    Returns
    -------
    client_fn(cid: str) -> fl.client.Client
    """
    soil_names = list(partitions.keys())

    def client_fn(cid: str) -> fl.client.Client:
        idx       = int(cid)
        soil_name = soil_names[idx]
        X, y      = partitions[soil_name]

        return AgriFlClient(
            client_id=idx + 1,
            soil_name=soil_name,
            X=X,
            y=y,
            num_classes=num_classes,
            local_epochs=local_epochs,
        ).to_client()

    return client_fn
