"""
hparam_search.py -- AgriFL ML
===============================
Manual grid search over DNN hyperparameters.

Searches over:
    learning_rate : [1e-4, 5e-4, 1e-3, 5e-3]
    batch_size    : [32, 64, 128, 256]
    hidden_units  : [64, 128, 256]          (first Dense layer)
    dropout       : [0.2, 0.3, 0.4]

For each combination trains the DNN for MAX_SEARCH_EPOCHS (with early stopping)
and records val_accuracy.  Best combination saved to:

    ml/models/best_params.json

Usage
-----
    cd d:/Krishi
    python ml/hparam_search.py

Output
------
    ml/models/best_params.json       best hyperparameter combination
    reports/metrics/hparam_results.csv   full grid-search log (sorted by val_acc)
"""

import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from federated.utils import load_raw, build_encoders
from federated.config import SEED, DATA_SUBPATH

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Search Space ──────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "batch_size":    [32, 64, 128, 256],
    "hidden_units":  [64, 128, 256],
    "dropout":       [0.2, 0.3, 0.4],
}

MAX_SEARCH_EPOCHS = 30    # max epochs per trial (early stopping cuts short)
PATIENCE          = 5     # early stopping patience
VAL_SPLIT         = 0.15  # fraction of training data used as validation


# ─────────────────────────────────────────────────────────────────────────────
# Model builder with tunable hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def build_tunable_dnn(
    input_dim: int,
    num_classes: int,
    hidden_units: int,
    dropout: float,
    learning_rate: float,
) -> keras.Model:
    """
    DNN with tunable first-layer width and dropout rate.

    Architecture:
        Input → Dense(hidden_units) → BN → Dropout(dropout)
                → Dense(hidden_units//2) → BN → Dropout(dropout)
                → Dense(hidden_units//4) → BN → Dropout(dropout*0.7)
                → Dense(num_classes, softmax)
    """
    h1 = max(hidden_units, 16)
    h2 = max(hidden_units // 2, 8)
    h3 = max(hidden_units // 4, 4)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(h1, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),

        layers.Dense(h2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),

        layers.Dense(h3, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout * 0.7),

        layers.Dense(num_classes, activation='softmax'),
    ], name='AgriFL_DNN_Tunable')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Grid Search
# ─────────────────────────────────────────────────────────────────────────────

def run_hparam_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    search_space: dict = None,
) -> dict:
    """
    Run a manual grid search over hyperparameters.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val     : validation data
    num_classes      : int  number of output classes
    search_space     : dict | None  — override SEARCH_SPACE if provided

    Returns
    -------
    dict : best hyperparameter combination + achieved val_accuracy
    """
    space   = search_space or SEARCH_SPACE
    keys    = list(space.keys())
    values  = list(space.values())
    combos  = list(itertools.product(*values))

    n_total  = len(combos)
    input_dim = X_train.shape[1]

    print(f"\n{'='*60}")
    print(f"  Hyperparameter Grid Search")
    print(f"  Total combinations : {n_total}")
    print(f"  Max epochs / trial : {MAX_SEARCH_EPOCHS}  (early stop @ patience={PATIENCE})")
    print(f"{'='*60}\n")

    results = []
    best_val_acc = -1.0
    best_params  = {}

    for trial_idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        print(f"  Trial {trial_idx:>3}/{n_total}  |  "
              f"lr={params['learning_rate']:.0e}  "
              f"bs={params['batch_size']:>3}  "
              f"hu={params['hidden_units']:>3}  "
              f"do={params['dropout']:.2f}",
              end="  →  ", flush=True)

        # Build fresh model
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        model = build_tunable_dnn(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_units=params['hidden_units'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )

        history = model.fit(
            X_train, y_train,
            epochs=MAX_SEARCH_EPOCHS,
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0,
        )

        # Best val accuracy achieved during this trial
        val_accs     = history.history.get('val_accuracy', [0])
        best_epoch   = int(np.argmax(val_accs))
        val_acc      = float(max(val_accs))
        val_loss     = float(history.history.get('val_loss', [0])[best_epoch])
        epochs_run   = len(val_accs)

        print(f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  "
              f"epochs={epochs_run}")

        row = {**params, "val_accuracy": val_acc, "val_loss": val_loss,
               "epochs_run": epochs_run, "trial": trial_idx}
        results.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = {**params, "val_accuracy": val_acc, "val_loss": val_loss}

        # Free memory
        del model
        keras.backend.clear_session()

    return best_params, results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_hparam_results(results_df: pd.DataFrame, save_path: str) -> None:
    """Bar chart of val_accuracy for every trial, sorted descending."""
    df_sorted = results_df.sort_values("val_accuracy", ascending=False).head(20)
    labels    = [
        f"lr={r['learning_rate']:.0e}\nbs={r['batch_size']}\nhu={r['hidden_units']}\ndo={r['dropout']:.1f}"
        for _, r in df_sorted.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(max(14, len(df_sorted) * 0.7), 5))
    bars = ax.bar(range(len(df_sorted)), df_sorted["val_accuracy"], color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_xlabel("Hyperparameter Combination")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Top-20 Hyperparameter Combinations (val_accuracy)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best bar
    bars[0].set_color('tomato')
    ax.axhline(df_sorted["val_accuracy"].max(), color='tomato',
               linestyle='--', linewidth=1, alpha=0.7, label='Best')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Hparam chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH   = os.path.join(BASE_DIR, DATA_SUBPATH)
    MODELS_DIR  = os.path.join(BASE_DIR, 'ml', 'models')
    METRICS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    raw_df = load_raw(DATA_PATH)
    X, y, encoders, scaler, feature_cols = build_encoders(raw_df.copy())
    num_classes = len(encoders['CROPS'].classes_)
    print(f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  Classes: {num_classes}")

    # 2. Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # 3. Run grid search
    best_params, all_results = run_hparam_search(
        X_train, y_train, X_val, y_val, num_classes
    )

    # 4. Save best_params.json
    best_path = os.path.join(MODELS_DIR, 'best_params.json')
    with open(best_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved → {best_path}")
    print(f"Best combination  : {best_params}")

    # 5. Save full results CSV
    results_df = pd.DataFrame(all_results).sort_values("val_accuracy", ascending=False)
    csv_path   = os.path.join(METRICS_DIR, 'hparam_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Full results saved → {csv_path}")

    # 6. Plot
    plot_hparam_results(results_df, save_path=os.path.join(METRICS_DIR, 'hparam_results.png'))

    # 7. Summary
    print(f"\n{'='*60}")
    print(f"  HYPERPARAMETER SEARCH COMPLETE")
    print(f"  Best val_accuracy : {best_params['val_accuracy']:.4f}")
    print(f"  learning_rate     : {best_params['learning_rate']}")
    print(f"  batch_size        : {best_params['batch_size']}")
    print(f"  hidden_units      : {best_params['hidden_units']}")
    print(f"  dropout           : {best_params['dropout']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
