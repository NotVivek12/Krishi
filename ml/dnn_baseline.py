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
    classification_report, confusion_matrix
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_and_preprocess(data_path: str):
    """Load raw CSV, clean, encode, and return X, y, encoders, scaler."""
    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # ── Exploratory checks ────────────────────────────────────────────────
    print(f"\nShape            : {df.shape}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    print(f"Duplicate rows   : {df.duplicated().sum()}")

    # ── Clean ─────────────────────────────────────────────────────────────
    df = df.drop_duplicates()
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # ── Drop leakage columns (confirmed from baseline analysis) ───────────
    leakage_cols = [
        'TYPE_OF_CROP', 'HARVESTED',
        'SOIL_PH_HIGH', 'CROPDURATION_MAX', 'MAX_TEMP',
        'WATERREQUIRED_MAX', 'RELATIVE_HUMIDITY_MAX',
        'N_MAX', 'P_MAX', 'K_MAX',
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    print(f"\nDropped {len(leakage_cols)} leakage columns.")
    print(f"Remaining features: {list(df.drop(columns='CROPS').columns)}")

    # ── Encode categoricals ───────────────────────────────────────────────
    cat_cols = ['SOIL', 'SOWN', 'WATER_SOURCE', 'SEASON']
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # ── Encode target ─────────────────────────────────────────────────────
    target_le = LabelEncoder()
    df['CROPS'] = target_le.fit_transform(df['CROPS'].astype(str))
    encoders['CROPS'] = target_le

    X = df.drop(columns=['CROPS']).values
    y = df['CROPS'].values

    # ── Scale features (critical for neural networks) ─────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, encoders, scaler, df.drop(columns=['CROPS']).columns.tolist()


def build_dnn(input_dim: int, num_classes: int) -> keras.Model:
    """Build the Dense Neural Network.

    Architecture
    ────────────
    Input(input_dim)
      → Dense(128, relu) → BatchNorm → Dropout(0.3)
      → Dense(64,  relu) → BatchNorm → Dropout(0.3)
      → Dense(32,  relu) → BatchNorm → Dropout(0.2)
      → Dense(num_classes, softmax)
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
    ], name='AgriFL_DNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def plot_training_history(history, save_path: str):
    """Plot and save training vs validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('DNN — Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('DNN — Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved → {save_path}")


def plot_confusion_matrix(cm, class_names, save_path: str):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Purples', fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('DNN — Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def save_comparison_csv(rf_path: str, dnn_metrics: dict, save_path: str):
    """Append DNN results to baseline_results.csv for side-by-side comparison."""
    rows = []

    if os.path.exists(rf_path):
        rf_df = pd.read_csv(rf_path)
        rows.append(rf_df.iloc[0].to_dict())

    rows.append({
        'Model': 'Dense Neural Network',
        'n_estimators': 'N/A',
        'Features Used': dnn_metrics['features'],
        'Leakage Columns Dropped': 10,
        'Train Size': dnn_metrics['train_size'],
        'Test Size': dnn_metrics['test_size'],
        'Accuracy': round(dnn_metrics['accuracy'], 4),
        'Precision (weighted)': round(dnn_metrics['precision'], 4),
        'Recall (weighted)': round(dnn_metrics['recall'], 4),
        'F1 Score (weighted)': round(dnn_metrics['f1'], 4),
        'Note': 'Dense(128→64→32→57) with BatchNorm + Dropout. StandardScaler applied.',
    })

    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"\nComparison CSV saved → {save_path}")


def main():
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path  = os.path.join(base_dir, 'data', 'raw', 'primary.csv')
    models_dir = os.path.join(base_dir, 'ml', 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    metrics_dir = os.path.join(reports_dir, 'metrics')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ── 1. Load & preprocess ──────────────────────────────────────────────
    X, y, encoders, scaler, feature_names = load_and_preprocess(data_path)
    num_classes = len(encoders['CROPS'].classes_)
    print(f"\nClasses : {num_classes}  |  Features : {X.shape[1]}")

    # ── 2. Train/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

    # ── 3. Build model ────────────────────────────────────────────────────
    model = build_dnn(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()

    # ── 4. Train ──────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5
        ),
    ]

    print("\nTraining DNN...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("\nEvaluating...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n{'--'*20}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'--'*20}")

    class_names = encoders['CROPS'].classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # ── 6. Plots ──────────────────────────────────────────────────────────
    plot_training_history(
        history,
        save_path=os.path.join(metrics_dir, 'dnn_training_curves.png')
    )

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(reports_dir, 'dnn_confusion_matrix.png')
    )

    # ── 7. Save model, encoders, scaler ───────────────────────────────────
    model_path = os.path.join(models_dir, 'dnn_model.keras')
    model.save(model_path)
    print(f"\nModel saved → {model_path}")

    joblib.dump(encoders, os.path.join(models_dir, 'dnn_encoders.pkl'))
    joblib.dump(scaler,   os.path.join(models_dir, 'dnn_scaler.pkl'))
    print(f"Encoders saved → {models_dir}/dnn_encoders.pkl")
    print(f"Scaler   saved → {models_dir}/dnn_scaler.pkl")

    # ── 8. Comparison CSV ─────────────────────────────────────────────────
    dnn_metrics = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'features': X.shape[1],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
    }
    save_comparison_csv(
        rf_path=os.path.join(reports_dir, 'baseline_results.csv'),
        dnn_metrics=dnn_metrics,
        save_path=os.path.join(reports_dir, 'baseline_results.csv'),
    )

    print("\nDNN pipeline complete.")


if __name__ == "__main__":
    main()
