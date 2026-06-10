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
