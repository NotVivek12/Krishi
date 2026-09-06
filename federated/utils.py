"""
utils.py -- AgriFL Federated Learning
======================================
Shared preprocessing and soil-based Non-IID data partitioning.

Each client represents a geographic region with a dominant soil type,
creating a realistic Non-IID federated learning scenario.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


# Confirmed leakage columns from baseline analysis
LEAKAGE_COLS = [
    'TYPE_OF_CROP', 'HARVESTED',
    'SOIL_PH_HIGH', 'CROPDURATION_MAX', 'MAX_TEMP',
    'WATERREQUIRED_MAX', 'RELATIVE_HUMIDITY_MAX',
    'N_MAX', 'P_MAX', 'K_MAX',
]

# Non-IID partition: one client per soil type (discovered dynamically)
# Override order to match project proposal if those soils exist
PREFERRED_SOIL_ORDER = [
    'Black soil', 'Red soil', 'Alluvial soil', 'Sandy soil', 'Loamy soil',
    'Clay soil', 'Sandy loam', 'Laterite soil', 'Saline soil',
]


def load_raw(data_path: str) -> pd.DataFrame:
    """Load CSV and perform basic cleaning."""
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    return df


def build_encoders(df: pd.DataFrame):
    """Fit encoders and scaler on the full dataset, return encoded arrays."""
    cat_cols = ['SOIL', 'SOWN', 'WATER_SOURCE', 'SEASON']
    encoders = {}

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    target_le = LabelEncoder()
    df['CROPS'] = target_le.fit_transform(df['CROPS'].astype(str))
    encoders['CROPS'] = target_le

    feature_cols = [c for c in df.columns if c != 'CROPS']
    X = df[feature_cols].values.astype(np.float32)
    y = df['CROPS'].values.astype(np.int32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, encoders, scaler, feature_cols


def partition_by_soil(raw_df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    """
    Partition dataset by SOIL type for Non-IID federated simulation.

    Returns a dict: { soil_name: (X_client, y_client) }

    This creates a realistic Non-IID setting because:
    - Each client (farm region) has data only from its local soil type
    - Crop distributions differ significantly between soil types
    - A model trained on one client's data won't generalize to others
    """
    # Get unique soil types present in the dataset
    soil_col_raw = raw_df['SOIL'].values
    unique_soils = sorted(raw_df['SOIL'].unique())

    # Re-order to match preferred order from proposal (if available)
    ordered = [s for s in PREFERRED_SOIL_ORDER if s in unique_soils]
    remaining = [s for s in unique_soils if s not in ordered]
    final_order = ordered + remaining

    print(f"\nDiscovered {len(final_order)} soil types for Non-IID partitioning:")
    partitions = {}
    for i, soil in enumerate(final_order):
        mask = soil_col_raw == soil
        X_s = X[mask]
        y_s = y[mask]
        partitions[soil] = (X_s, y_s)
        print(f"  Client {i+1}: {soil:20s} -> {X_s.shape[0]:5d} samples, "
              f"{len(np.unique(y_s))} unique crops")

    return partitions


def build_dnn(input_dim: int, num_classes: int) -> keras.Model:
    """
    Shared DNN architecture used by every FL client.
    All clients must use the same architecture for FedAvg weight averaging.

    Input -> Dense(128) -> Dense(64) -> Dense(32) -> Output(num_classes)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax'),
    ], name='AgriFL_FL_DNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_model_weights(model: keras.Model):
    """Extract model weights as a list of numpy arrays."""
    return model.get_weights()


def set_model_weights(model: keras.Model, weights):
    """Apply a list of numpy arrays as model weights."""
    model.set_weights(weights)


def load_processed_splits(processed_dir: str):
    """Load generated train/test splits and return arrays plus metadata."""
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [
        filename
        for filename in required
        if not os.path.exists(os.path.join(processed_dir, filename))
    ]
    if missing:
        raise FileNotFoundError(
            "Processed split files are missing: " + ", ".join(missing)
        )

    X_train_df = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test_df = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train_df = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))
    y_test_df = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))

    feature_cols = [col for col in X_train_df.columns if col != "source_row_id"]
    return {
        "X_train": X_train_df[feature_cols].to_numpy(dtype=np.float32),
        "X_test": X_test_df[feature_cols].to_numpy(dtype=np.float32),
        "y_train": y_train_df["target"].to_numpy(dtype=np.int32),
        "y_test": y_test_df["target"].to_numpy(dtype=np.int32),
        "feature_cols": feature_cols,
        "num_classes": int(y_train_df["target"].nunique()),
    }


def load_client_shards(shards_root: str, n_clients: int, val_fraction: float = 0.2):
    """
    Load generated training-only client shards for FL.

    Each shard CSV contains scaled/encoded model features plus metadata columns.
    A local train/validation split is made inside each client from that shard
    only; the global X_test/y_test split remains isolated.
    """
    shard_dir = os.path.join(shards_root, f"{n_clients}_clients")
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(f"Client shard directory not found: {shard_dir}")

    shard_files = sorted(
        filename
        for filename in os.listdir(shard_dir)
        if filename.startswith("client_") and filename.endswith(".csv")
    )
    if len(shard_files) != n_clients:
        raise ValueError(
            f"Expected {n_clients} shard files in {shard_dir}, "
            f"found {len(shard_files)}."
        )

    clients = []
    for index, filename in enumerate(shard_files, start=1):
        frame = pd.read_csv(os.path.join(shard_dir, filename))
        feature_cols = [
            col
            for col in frame.columns
            if col not in {"source_row_id", "target", "crop_label", "SOIL_raw"}
        ]
        X = frame[feature_cols].to_numpy(dtype=np.float32)
        y = frame["target"].to_numpy(dtype=np.int32)
        stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=val_fraction,
            random_state=42,
            stratify=stratify,
        )
        soil_counts = frame["SOIL_raw"].value_counts()
        dominant_soil = str(soil_counts.index[0])
        clients.append(
            {
                "id": index,
                "soil": dominant_soil,
                "X_train": X_tr,
                "y_train": y_tr,
                "X_val": X_val,
                "y_val": y_val,
                "n_train": len(X_tr),
                "source_file": filename,
                "soil_distribution": soil_counts.to_dict(),
            }
        )

    feature_count = len(feature_cols) if shard_files else 0
    num_classes = int(
        max(client["y_train"].max() for client in clients) + 1
    )
    return clients, feature_count, num_classes
