"""
versioning.py -- AgriFL Federated Learning
============================================
Per-round global model versioning and metadata tracking.

Saves a snapshot of the global model after every FL round so you can:
  - Roll back to any earlier round
  - Plot accuracy/loss evolution with matched model checkpoints
  - Compare model drift across rounds

Directory layout
----------------
    ml/models/versioned/
    ├── round_01.keras
    ├── round_02.keras
    ├── ...
    └── metadata.json      ← list of { round, accuracy, loss, timestamp, path }

Usage
-----
    from ml.versioning import ModelVersioner

    versioner = ModelVersioner(base_dir="d:/Krishi")
    versioner.save_round(model=global_model, round_num=1,
                         metrics={"accuracy": 0.82, "loss": 0.41})

    versions = versioner.list_versions()
    best     = versioner.best_round(metric="accuracy")
    model    = versioner.load_round(round_num=best["round"])
"""

import os
import json
import datetime

import numpy as np


class ModelVersioner:
    """
    Manages versioned checkpoints of the global FL model.

    Parameters
    ----------
    base_dir : str
        Root of the project (e.g. "d:/Krishi").  Versioned models are saved
        under <base_dir>/ml/models/versioned/.
    experiment_tag : str
        Optional label prepended to filenames (e.g. "10_clients").
        Useful when running multiple experiments with different NUM_CLIENTS.
    """

    def __init__(self, base_dir: str, experiment_tag: str = ""):
        self.base_dir        = base_dir
        self.experiment_tag  = experiment_tag
        self.versioned_dir   = os.path.join(base_dir, "ml", "models", "versioned")
        self.metadata_path   = os.path.join(self.versioned_dir, "metadata.json")
        os.makedirs(self.versioned_dir, exist_ok=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _model_filename(self, round_num: int) -> str:
        tag    = f"{self.experiment_tag}_" if self.experiment_tag else ""
        return f"{tag}round_{round_num:02d}.keras"

    def _load_metadata(self) -> list:
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return []

    def _save_metadata(self, records: list) -> None:
        with open(self.metadata_path, "w") as f:
            json.dump(records, f, indent=2, default=str)

    # ── Public API ────────────────────────────────────────────────────────────

    def save_round(self, model, round_num: int, metrics: dict) -> str:
        """
        Save the global model for `round_num` and update metadata.

        Parameters
        ----------
        model     : keras.Model  — the current global model
        round_num : int          — current FL round (1-indexed)
        metrics   : dict         — e.g. {"accuracy": 0.82, "loss": 0.41}

        Returns
        -------
        str : absolute path to the saved .keras file
        """
        filename  = self._model_filename(round_num)
        save_path = os.path.join(self.versioned_dir, filename)

        model.save(save_path)

        records = self._load_metadata()

        # Remove any existing entry for this round (allows re-runs)
        records = [r for r in records if r.get("round") != round_num
                   or r.get("experiment_tag") != self.experiment_tag]

        records.append({
            "round":          round_num,
            "experiment_tag": self.experiment_tag,
            "accuracy":       round(float(metrics.get("accuracy", 0)), 4),
            "loss":           round(float(metrics.get("loss", 0)), 4),
            "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
            "filename":       filename,
            "path":           save_path,
        })

        # Keep records sorted by round number
        records.sort(key=lambda r: (r.get("experiment_tag", ""), r["round"]))
        self._save_metadata(records)

        return save_path

    def load_round(self, round_num: int):
        """
        Load and return the saved Keras model for `round_num`.

        Raises FileNotFoundError if no snapshot exists for that round.
        """
        import tensorflow as tf  # lazy import — only needed when loading

        filename  = self._model_filename(round_num)
        load_path = os.path.join(self.versioned_dir, filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"No versioned model found for round {round_num} "
                f"(expected: {load_path})"
            )

        return tf.keras.models.load_model(load_path)

    def list_versions(self) -> list:
        """
        Return all saved version records for this experiment tag.

        Returns
        -------
        list of dict  sorted by round number
        """
        records = self._load_metadata()
        if self.experiment_tag:
            records = [r for r in records
                       if r.get("experiment_tag") == self.experiment_tag]
        return records

    def best_round(self, metric: str = "accuracy") -> dict:
        """
        Return the metadata record for the round with the best metric value.

        Parameters
        ----------
        metric : str  — "accuracy" (higher is better) or "loss" (lower is better)

        Returns
        -------
        dict : the metadata record for the best round
        """
        records = self.list_versions()
        if not records:
            raise ValueError("No versioned models saved yet.")

        if metric == "accuracy":
            return max(records, key=lambda r: r.get("accuracy", 0))
        elif metric == "loss":
            return min(records, key=lambda r: r.get("loss", float("inf")))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'accuracy' or 'loss'.")

    def print_summary(self) -> None:
        """Print a formatted table of all saved versions."""
        records = self.list_versions()
        if not records:
            print("No versioned models found.")
            return

        tag = f" [{self.experiment_tag}]" if self.experiment_tag else ""
        print(f"\nVersioned Models{tag}")
        print(f"{'Round':>6}  {'Accuracy':>10}  {'Loss':>10}  {'Timestamp':<20}  Filename")
        print("-" * 72)
        for r in records:
            print(
                f"  {r['round']:>4}  {r['accuracy']:>10.4f}  {r['loss']:>10.4f}  "
                f"{r['timestamp']:<20}  {r['filename']}"
            )
