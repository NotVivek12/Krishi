"""
federated_train.py -- AgriFL
==============================
Main entry point for the Federated Learning simulation.

Uses a lightweight sequential simulation loop (no Ray) to avoid
Windows memory issues. Each round: clients train locally, server
aggregates weights via FedAvg, global model is updated.

Non-IID Setup: clients are assigned one dominant soil type each.
Privacy: only model weights are exchanged, never raw data.

Usage
-----
    cd d:/Krishi

    # Default (5 clients):
    python federated/federated_train.py

    # Custom client count:
    python federated/federated_train.py --clients 10
    python federated/federated_train.py --clients 20

    # Disable straggler simulation (production):
    python federated/federated_train.py --clients 5 --no-stragglers

Output
------
    reports/
    |- fl_5_clients.csv            per-round accuracy + loss (per run)
    +- metrics/
       +- fl_training_curves_5_clients.png

    ml/models/
    |- fl_global_model_5_clients.keras   final global model
    +- versioned/
       |- 5_clients_round_01.keras
       +- metadata.json
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from sklearn.model_selection import train_test_split

from federated.config import (
    NUM_ROUNDS, LOCAL_EPOCHS, NUM_CLIENTS, SEED, BATCH_SIZE,
    STRAGGLER_TIMEOUT, STRAGGLER_SLOW_PROB, STRAGGLER_SLOW_MULT,
    MIN_CLIENTS_FOR_AGG, SAVE_ROUND_MODELS,
)
from federated.aggregation import fedavg
from federated.utils import (
    load_raw, build_encoders, build_dnn,
    get_model_weights, set_model_weights,
)
from ml.straggler import StragglerSimulator, filter_stragglers, straggler_summary
from ml.versioning import ModelVersioner

# -- Reproducibility -----------------------------------------------------------
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -- Paths ---------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'primary.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'ml', 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
METRICS_DIR = os.path.join(REPORTS_DIR, 'metrics')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Non-IID Soil Partitioning
# -----------------------------------------------------------------------------

def partition_by_soil(raw_df, X, y, n_clients=NUM_CLIENTS):
    """
    Select the top-N soil types by sample count and assign one client each.
    Returns list of dicts with client metadata and local train/val splits.
    """
    top_soils = (
        raw_df['SOIL'].value_counts()
        .head(n_clients)
        .index.tolist()
    )

    clients = []
    print(f"\nNon-IID Partition ({n_clients} clients):")
    print(f"{'Client':<10} {'Soil Type':<35} {'Samples':>8} {'Crops':>6}")
    print("-" * 65)

    for i, soil in enumerate(top_soils):
        mask = raw_df['SOIL'].values == soil
        X_s  = X[mask]
        y_s  = y[mask]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_s, y_s, test_size=0.2, random_state=SEED,
            stratify=y_s if len(np.unique(y_s)) > 1 else None,
        )

        clients.append({
            'id':       i + 1,
            'soil':     soil,
            'X_train':  X_tr,
            'y_train':  y_tr,
            'X_val':    X_val,
            'y_val':    y_val,
            'n_train':  len(X_tr),
        })

        print(f"Client {i+1:<4}  {soil:<35} {len(X_s):>8} {len(np.unique(y_s)):>6}")

    return clients


# -----------------------------------------------------------------------------
# Sequential FL Simulation Loop
# -----------------------------------------------------------------------------

def run_simulation(clients, global_model, num_rounds, versioner=None, straggler_sim=None):
    """
    Run the FL simulation sequentially (no Ray/gRPC needed).

    Each round:
      1. Broadcast global weights to all clients
      2. Each client trains locally (stragglers may be skipped)
      3. Collect updated weights + dataset sizes from responding clients
      4. Aggregate via FedAvg -> new global weights
      5. Evaluate global model on each client val set
      6. (Optional) Save per-round model snapshot
      7. Log round metrics
    """
    best_acc     = -1.0
    best_weights = None
    best_round   = -1

    for rnd in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  Round {rnd}/{num_rounds}")
        print(f"{'='*60}")

        global_weights = get_model_weights(global_model)
        raw_results    = []

        # -- Local Training ---------------------------------------------------
        for client in clients:
            local_model = build_dnn(
                input_dim=client['X_train'].shape[1],
                num_classes=global_model.output_shape[-1],
            )
            set_model_weights(local_model, global_weights)

            if straggler_sim is not None:
                result = straggler_sim.train_client(
                    client, local_model, LOCAL_EPOCHS, BATCH_SIZE
                )
            else:
                local_model.fit(
                    client['X_train'], client['y_train'],
                    epochs=LOCAL_EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                )
                result = (get_model_weights(local_model), client['n_train'])

            raw_results.append(result)
            status = "OK" if result is not None else "STRAGGLER"
            print(f"  Client {client['id']} [{client['soil'][:30]}]"
                  f" -- {client['n_train']} samples  [{status}]")

        # -- Filter Stragglers ------------------------------------------------
        if straggler_sim is not None:
            client_weights, client_sizes, n_stragglers = filter_stragglers(raw_results)
            if n_stragglers > 0:
                print(f"\n  {straggler_summary(len(clients), n_stragglers)}")
        else:
            client_weights = [r[0] for r in raw_results]
            client_sizes   = [r[1] for r in raw_results]
            n_stragglers   = 0

        if len(client_weights) < MIN_CLIENTS_FOR_AGG:
            print(f"\n  [WARN] Only {len(client_weights)} clients responded "
                  f"(min required: {MIN_CLIENTS_FOR_AGG}). Skipping round.")
            continue

        # -- FedAvg Aggregation -----------------------------------------------
        new_global_weights = fedavg(client_weights, client_sizes)
        set_model_weights(global_model, new_global_weights)

        # -- Global Evaluation ------------------------------------------------
        total_correct = 0
        total_samples = 0
        total_loss    = 0.0
        print()

        for client in clients:
            loss, acc = global_model.evaluate(
                client['X_val'], client['y_val'], verbose=0
            )
            n = len(client['y_val'])
            total_correct += acc * n
            total_samples += n
            total_loss    += loss * n
            print(f"  [Client {client['id']} | {client['soil'][:28]}]  "
                  f"val_loss={loss:.4f}  val_acc={acc:.4f}")

        weighted_acc  = total_correct / total_samples
        weighted_loss = total_loss    / total_samples

        # -- Track best round -------------------------------------------------
        is_best = weighted_acc > best_acc
        if is_best:
            best_acc     = weighted_acc
            best_weights = [w.copy() for w in get_model_weights(global_model)]
            best_round   = rnd

        best_marker = " *** BEST ***" if is_best else ""
        print(f"\n  >> Global weighted accuracy: {weighted_acc:.4f}  "
              f"loss: {weighted_loss:.4f}"
              f"  (stragglers skipped: {n_stragglers}){best_marker}")

        metrics = {
            'round':        rnd,
            'accuracy':     round(weighted_acc, 4),
            'loss':         round(weighted_loss, 4),
            'stragglers':   n_stragglers,
            'clients_used': len(client_weights),
            'is_best':      is_best,
        }
        round_history.append(metrics)

        # -- Per-round Versioning ---------------------------------------------
        if versioner is not None and SAVE_ROUND_MODELS:
            saved = versioner.save_round(global_model, rnd, metrics)
            print(f"  Snapshot saved -> {os.path.basename(saved)}")

    return round_history, best_weights, best_round


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_fl_curves(history, save_path, n_clients):
    rounds = [h['round']    for h in history]
    accs   = [h['accuracy'] for h in history]
    losses = [h['loss']     for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(rounds, accs, marker='o', color='steelblue', linewidth=2)
    axes[0].set_title(f'AgriFL -- Global Accuracy per Round ({n_clients} clients)')
    axes[0].set_xlabel('Communication Round')
    axes[0].set_ylabel('Weighted Avg Accuracy')
    axes[0].set_xticks(rounds)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, losses, marker='o', color='tomato', linewidth=2)
    axes[1].set_title(f'AgriFL -- Global Loss per Round ({n_clients} clients)')
    axes[1].set_xlabel('Communication Round')
    axes[1].set_ylabel('Weighted Avg Loss')
    axes[1].set_xticks(rounds)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nFL training curves saved -> {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AgriFL Federated Learning Simulation")
    parser.add_argument(
        '--clients', type=int, default=NUM_CLIENTS,
        help=f'Number of FL clients / soil partitions (default: {NUM_CLIENTS})'
    )
    parser.add_argument(
        '--rounds', type=int, default=NUM_ROUNDS,
        help=f'Number of FL communication rounds (default: {NUM_ROUNDS})'
    )
    parser.add_argument(
        '--epochs', type=int, default=LOCAL_EPOCHS,
        help=f'Local training epochs per client per round (default: {LOCAL_EPOCHS})'
    )
    parser.add_argument(
        '--no-stragglers', action='store_true',
        help='Disable straggler simulation (all clients always complete)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    n_clients      = args.clients
    n_rounds       = args.rounds
    local_epochs   = args.epochs
    use_stragglers = not args.no_stragglers

    global LOCAL_EPOCHS
    LOCAL_EPOCHS = local_epochs

    tag = f"{n_clients}_clients"

    print("=" * 60)
    print("  AgriFL -- Federated Learning Simulation")
    print(f"  Strategy      : FedAvg")
    print(f"  Rounds        : {n_rounds}")
    print(f"  Local Epochs  : {LOCAL_EPOCHS}")
    print(f"  Clients       : {n_clients} (Non-IID by soil type)")
    print(f"  Stragglers    : {'enabled (sim)' if use_stragglers else 'disabled'}")
    print("=" * 60)

    # 1. Load and preprocess
    print("\nLoading and preprocessing data...")
    raw_df = load_raw(DATA_PATH)
    X, y, encoders, scaler, feature_cols = build_encoders(raw_df.copy())
    num_classes = len(encoders['CROPS'].classes_)
    n_features  = X.shape[1]
    print(f"Total samples: {X.shape[0]}  |  Features: {n_features}  |  Classes: {num_classes}")

    # 2. Non-IID partitioning
    clients = partition_by_soil(raw_df, X, y, n_clients=n_clients)

    # 3. Initialize global model
    global_model = build_dnn(input_dim=n_features, num_classes=num_classes)
    global_model.predict(X[:1], verbose=0)

    # 4. Set up versioner and straggler simulator
    versioner = ModelVersioner(base_dir=BASE_DIR, experiment_tag=tag)

    straggler_sim = None
    if use_stragglers:
        straggler_sim = StragglerSimulator(
            timeout_seconds=STRAGGLER_TIMEOUT,
            slow_client_prob=STRAGGLER_SLOW_PROB,
            slow_multiplier=STRAGGLER_SLOW_MULT,
            seed=SEED,
        )

    # 5. Run FL simulation
    round_history, best_weights, best_rnd = run_simulation(
        clients, global_model,
        num_rounds=n_rounds,
        versioner=versioner,
        straggler_sim=straggler_sim,
    )

    # 6. Save training curves
    curves_path = os.path.join(METRICS_DIR, f'fl_training_curves_{tag}.png')
    plot_fl_curves(round_history, save_path=curves_path, n_clients=n_clients)

    # 7. Save per-experiment CSV
    results_df   = pd.DataFrame(round_history)
    results_path = os.path.join(REPORTS_DIR, f'fl_{tag}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"FL results saved -> {results_path}")

    # 8. Save BEST global model (best-performing round, not final round)
    best_model_path = os.path.join(MODELS_DIR, f'fl_best_model_{tag}.keras')
    set_model_weights(global_model, best_weights)   # restore best weights
    global_model.save(best_model_path)
    print(f"Best model saved  -> {best_model_path}  (round {best_rnd}, "
          f"acc={max(h['accuracy'] for h in round_history):.4f})")

    # 9. Also save the final-round model for reference
    final_model_path = os.path.join(MODELS_DIR, f'fl_global_model_{tag}.keras')
    # reload final weights from versioner snapshot if available
    try:
        final_snap = versioner.load_round(n_rounds)
        final_snap.save(final_model_path)
    except Exception:
        # fallback: current model weights (already set to best above)
        global_model.save(final_model_path)
    print(f"Final model saved -> {final_model_path}")

    # 10. Save encoders / scaler
    joblib.dump(encoders, os.path.join(MODELS_DIR, f'fl_encoders_{tag}.pkl'))
    joblib.dump(scaler,   os.path.join(MODELS_DIR, f'fl_scaler_{tag}.pkl'))
    print(f"Encoders / Scaler saved -> {MODELS_DIR}")

    # 11. Print version summary
    if SAVE_ROUND_MODELS:
        versioner.print_summary()

    # 12. Final summary
    best_h = max(round_history, key=lambda h: h['accuracy'])
    final  = round_history[-1]
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS -- {tag.replace('_', ' ').title()}")
    print(f"  Last round      : {final['round']}  acc={final['accuracy']:.4f}  loss={final['loss']:.4f}")
    print(f"  Best round      : {best_h['round']}  acc={best_h['accuracy']:.4f}  loss={best_h['loss']:.4f}")
    print(f"  Best model saved: fl_best_model_{tag}.keras")
    print(f"{'='*60}")
    print("\nFederated Learning simulation complete.")


if __name__ == "__main__":
    main()