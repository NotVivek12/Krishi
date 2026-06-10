"""
cnn_lstm_model.py -- AgriFL
============================
Hybrid CNN-LSTM model for crop classification on tabular data.

Since the dataset is tabular (not sequential), each feature is treated as a
single time-step channel: X is reshaped from (samples, features) to
(samples, features, 1) so Conv1D can extract local feature patterns before
the LSTM captures cross-feature dependencies.

Architecture
------------
  Input(samples, features, 1)
    -> Conv1D(64, kernel=3, relu) -> MaxPooling1D(2)
    -> Conv1D(32, kernel=3, relu) -> MaxPooling1D(2)
    -> LSTM(64, return_sequences=False)
    -> Dense(64, relu) -> Dropout(0.3)
    -> Dense(num_classes, softmax)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_and_preprocess(data_path: str):
    """Load CSV, remove leakage columns, encode, scale, and return tensors."""
    print("Loading dataset...")
    df = pd.read_csv(data_path)

    print(f"Shape           : {df.shape}")
    print(f"Missing values  : {df.isnull().sum().sum()}")
    print(f"Duplicate rows  : {df.duplicated().sum()}")

    df = df.drop_duplicates()
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # Confirmed leakage columns (each has exactly 1 unique value per crop)
    leakage_cols = [
        'TYPE_OF_CROP', 'HARVESTED',
        'SOIL_PH_HIGH', 'CROPDURATION_MAX', 'MAX_TEMP',
        'WATERREQUIRED_MAX', 'RELATIVE_HUMIDITY_MAX',
        'N_MAX', 'P_MAX', 'K_MAX',
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    print(f"Dropped {len(leakage_cols)} leakage columns. Remaining features: {df.shape[1] - 1}")

    # Encode categorical features
    cat_cols = ['SOIL', 'SOWN', 'WATER_SOURCE', 'SEASON']
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    df['CROPS'] = target_le.fit_transform(df['CROPS'].astype(str))
    encoders['CROPS'] = target_le

    feature_names = [c for c in df.columns if c != 'CROPS']
    X = df[feature_names].values
    y = df['CROPS'].values

    # Scale (critical for CNN/LSTM stability)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, encoders, scaler, feature_names


def build_cnn_lstm(timesteps: int, num_classes: int) -> keras.Model:
    """
    Build CNN-LSTM hybrid model.

    Input shape: (batch, timesteps, 1)  -- each feature is one time step

    Conv1D(64)  -> MaxPooling1D(2)
    Conv1D(32)  -> MaxPooling1D(2)
    LSTM(64)
    Dense(64)   -> Dropout(0.3)
    Dense(num_classes, softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(timesteps, 1), name='input'),

        # CNN block 1 -- local feature extraction
        layers.Conv1D(filters=64, kernel_size=3, padding='same',
                      activation='relu', name='conv1'),
        layers.MaxPooling1D(pool_size=2, name='pool1'),

        # CNN block 2 -- higher-level patterns
        layers.Conv1D(filters=32, kernel_size=3, padding='same',
                      activation='relu', name='conv2'),
        layers.MaxPooling1D(pool_size=2, name='pool2'),

        # LSTM -- cross-feature dependencies
        layers.LSTM(units=64, name='lstm'),

        # Dense head
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output'),

    ], name='AgriFL_CNN_LSTM')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def plot_training_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('CNN-LSTM -- Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('CNN-LSTM -- Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved -> {save_path}")


def plot_confusion_matrix(cm, class_names, save_path: str):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Greens', fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('CNN-LSTM -- Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved -> {save_path}")


def update_comparison_csv(metrics: dict, csv_path: str):
    """Append CNN-LSTM row to the baseline_results.csv comparison table."""
    new_row = {
        'Model': 'CNN-LSTM',
        'n_estimators': 'N/A',
        'Features Used': metrics['features'],
        'Leakage Columns Dropped': 10,
        'Train Size': metrics['train_size'],
        'Test Size': metrics['test_size'],
        'Accuracy': round(metrics['accuracy'], 4),
        'Precision (weighted)': round(metrics['precision'], 4),
        'Recall (weighted)': round(metrics['recall'], 4),
        'F1 Score (weighted)': round(metrics['f1'], 4),
        'Note': 'Conv1D(64)->Pool->Conv1D(32)->Pool->LSTM(64)->Dense(64)->Output(57). Tabular X reshaped to (N, features, 1).',
    }

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[df_existing['Model'] != 'CNN-LSTM']
        df_out = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_out = pd.DataFrame([new_row])

    df_out.to_csv(csv_path, index=False)
    print(f"Comparison CSV updated -> {csv_path}")


def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path   = os.path.join(base_dir, 'data', 'raw', 'primary.csv')
    models_dir  = os.path.join(base_dir, 'ml', 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    metrics_dir = os.path.join(reports_dir, 'metrics')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Load & preprocess
    X, y, encoders, scaler, feature_names = load_and_preprocess(data_path)
    num_classes = len(encoders['CROPS'].classes_)
    n_features  = X.shape[1]
    print(f"\nClasses: {num_classes}  |  Features: {n_features}")

    # 2. Reshape for CNN-LSTM: (samples, features, 1)
    X_3d = X.reshape(X.shape[0], n_features, 1)
    print(f"Reshaped X: {X.shape} -> {X_3d.shape}")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_3d, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 4. Build & summarize model
    model = build_cnn_lstm(timesteps=n_features, num_classes=num_classes)
    model.summary()

    # 5. Train
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    print("\nTraining CNN-LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Evaluate
    print("\nEvaluating...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*40}")
    print(f"  Model     : CNN-LSTM")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*40}")

    class_names = encoders['CROPS'].classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # 7. Plots
    plot_training_history(
        history,
        save_path=os.path.join(metrics_dir, 'cnn_lstm_training_curves.png')
    )
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(reports_dir, 'cnn_lstm_confusion_matrix.png')
    )

    # 8. Save model artifacts
    model_path = os.path.join(models_dir, 'cnn_lstm_model.keras')
    model.save(model_path)
    print(f"\nModel saved    -> {model_path}")

    joblib.dump(encoders, os.path.join(models_dir, 'cnn_lstm_encoders.pkl'))
    joblib.dump(scaler,   os.path.join(models_dir, 'cnn_lstm_scaler.pkl'))
    print(f"Encoders saved -> {models_dir}/cnn_lstm_encoders.pkl")
    print(f"Scaler saved   -> {models_dir}/cnn_lstm_scaler.pkl")

    # 9. Update comparison CSV
    update_comparison_csv(
        metrics={
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'features': n_features,
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
        },
        csv_path=os.path.join(reports_dir, 'baseline_results.csv'),
    )

    print("\nCNN-LSTM pipeline complete.")


if __name__ == "__main__":
    main()
