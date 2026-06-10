"""
config.py -- AgriFL Federated Learning
========================================
Central configuration for all FL hyperparameters and constants.

Import from here instead of hard-coding values in individual modules:

    from federated.config import NUM_ROUNDS, LOCAL_EPOCHS, NUM_CLIENTS, SEED
"""

# ── Federated Learning ────────────────────────────────────────────────────────
NUM_ROUNDS   = 25    # number of communication rounds between server and clients
LOCAL_EPOCHS = 5     # local training epochs per client per round
NUM_CLIENTS  = 5     # default number of simulated farm regions (overridable via CLI)
SEED         = 42    # global random seed for reproducibility

# ── Local Training ────────────────────────────────────────────────────────────
BATCH_SIZE   = 64    # mini-batch size used during local client training

# ── Straggler Handling ────────────────────────────────────────────────────────
STRAGGLER_TIMEOUT    = 60.0   # seconds — clients slower than this are skipped
STRAGGLER_SLOW_PROB  = 0.0    # set > 0 (e.g. 0.2) to inject artificial slowness for testing
STRAGGLER_SLOW_MULT  = 3.0    # slowdown multiplier applied to simulated stragglers
MIN_CLIENTS_FOR_AGG  = 2      # minimum responding clients required to run aggregation

# ── Model Versioning ─────────────────────────────────────────────────────────
SAVE_ROUND_MODELS    = True   # if True, save global model snapshot after every round

# ── Paths (relative to project root) ─────────────────────────────────────────
DATA_SUBPATH    = "data/raw/primary.csv"
MODELS_SUBDIR   = "ml/models"
REPORTS_SUBDIR  = "reports"
METRICS_SUBDIR  = "reports/metrics"
VERSIONED_SUBDIR = "ml/models/versioned"
