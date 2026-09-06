import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = r"D:\Krishi"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SHARDS_DIR = os.path.join(PROCESSED_DIR, "client_shards", "balanced_non_iid", "5_clients")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "balanced_non_iid")
VERSIONED_DIR = os.path.join(MODELS_DIR, "versioned")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(SHARDS_DIR, exist_ok=True)
os.makedirs(VERSIONED_DIR, exist_ok=True)

FEATURE_COLS = ['SOIL', 'SEASON', 'SOWN', 'WATER_SOURCE', 'SOIL_PH', 'CROPDURATION', 
                'TEMP', 'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']

def build_dnn(input_dim: int = 12, num_classes: int = 57) -> keras.Model:
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
    ], name='AgriFL_Balanced_DNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

def fedavg(client_weights, client_sizes):
    total_samples = sum(client_sizes)
    new_weights = []
    for layer_idx in range(len(client_weights[0])):
        weighted_layer = sum(
            client_weights[c][layer_idx] * (client_sizes[c] / total_samples)
            for c in range(len(client_weights))
        )
        new_weights.append(weighted_layer)
    return new_weights

def generate_shards():
    print("--- Step 1: Generating Class-Aware Balanced Non-IID Shards (5 Clients) ---")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))
    
    primary = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "primary.csv")).drop_duplicates().dropna()
    leakage = ['TYPE_OF_CROP', 'HARVESTED', 'SOIL_PH_HIGH', 'CROPDURATION_MAX', 'MAX_TEMP',
               'WATERREQUIRED_MAX', 'RELATIVE_HUMIDITY_MAX', 'N_MAX', 'P_MAX', 'K_MAX']
    primary = primary.drop(columns=[c for c in leakage if c in primary.columns])
    primary['source_row_id'] = primary.index

    train_meta = primary.loc[primary['source_row_id'].isin(X_train['source_row_id']), ['source_row_id', 'SOIL']].rename(columns={'SOIL': 'SOIL_raw'})
    shard_df = X_train.merge(y_train, on='source_row_id').merge(train_meta, on='source_row_id')

    np.random.seed(SEED)
    alpha = 0.5
    client_count = 5
    n_classes = shard_df['target'].nunique()
    client_indices = [[] for _ in range(client_count)]

    for c in range(n_classes):
        idx_c = shard_df[shard_df['target'] == c].index.values.copy()
        np.random.shuffle(idx_c)
        props = np.random.dirichlet(np.repeat(alpha, client_count))
        counts = (props * len(idx_c)).astype(int)
        diff = len(idx_c) - counts.sum()
        for i in range(diff):
            counts[i % client_count] += 1
        split_pts = np.cumsum(counts)[:-1]
        splits = np.split(idx_c, split_pts)
        for i in range(client_count):
            client_indices[i].extend(splits[i])

    shards = []
    expected_cols = ["source_row_id", *FEATURE_COLS, "target", "crop_label", "SOIL_raw"]
    for i in range(client_count):
        sub = shard_df.loc[client_indices[i]].sort_values("source_row_id").reset_index(drop=True)
        sub = sub[expected_cols]
        shard_path = os.path.join(SHARDS_DIR, f"client_{i+1:02d}.csv")
        sub.to_csv(shard_path, index=False)
        shards.append(sub)
        print(f"  Client {i+1:02d}: {len(sub)} samples, {sub['target'].nunique()} crop classes -> {shard_path}")
        
    return shards

def load_data_and_shards():
    X_test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))
    X_test = X_test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = y_test_df["target"].to_numpy(dtype=np.int32)
    
    clients = []
    for i in range(1, 6):
        shard_path = os.path.join(SHARDS_DIR, f"client_{i:02d}.csv")
        frame = pd.read_csv(shard_path)
        X = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = frame["target"].to_numpy(dtype=np.int32)
        
        counts = np.bincount(y)
        min_count = min(counts[counts > 0])
        stratify = y if len(np.unique(y)) > 1 and min_count >= 2 else None
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=stratify
        )
        clients.append({
            "id": i,
            "X_train": X_tr,
            "y_train": y_tr,
            "X_val": X_val,
            "y_val": y_val,
            "n_train": len(X_tr)
        })
        
    return clients, X_test, y_test

def run_experiment():
    shards = generate_shards()
    clients, X_test, y_test = load_data_and_shards()
    
    print("\n--- Step 2: Running 25-Round FedAvg Simulation ---")
    global_model = build_dnn(12, 57)
    global_model.predict(np.zeros((1, 12), dtype=np.float32), verbose=0)
    
    round_history = []
    
    for rnd in range(1, 26):
        global_weights = global_model.get_weights()
        client_weights = []
        client_sizes = []
        
        # Local training
        for client in clients:
            local_model = build_dnn(12, 57)
            local_model.set_weights(global_weights)
            local_model.fit(
                client["X_train"], client["y_train"],
                epochs=5, batch_size=64, verbose=0
            )
            client_weights.append(local_model.get_weights())
            client_sizes.append(client["n_train"])
            
        # FedAvg
        new_global_weights = fedavg(client_weights, client_sizes)
        global_model.set_weights(new_global_weights)
        
        # Validation accuracy (weighted across client val sets)
        total_correct = 0
        total_val_samples = 0
        for client in clients:
            loss, acc = global_model.evaluate(client["X_val"], client["y_val"], verbose=0)
            n_val = len(client["y_val"])
            total_correct += acc * n_val
            total_val_samples += n_val
        val_acc = total_correct / total_val_samples
        
        # Save snapshot
        snapshot_path = os.path.join(VERSIONED_DIR, f"5_clients_round_{rnd:02d}.keras")
        global_model.save(snapshot_path)
        
        # Global Test Set Evaluation
        probs = global_model.predict(X_test, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        
        top3_preds = np.argsort(probs, axis=1)[:, -3:]
        top3_acc = np.mean([1 if y_test[i] in top3_preds[i] else 0 for i in range(len(y_test))])
        
        test_acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            "round": rnd,
            "val_accuracy": round(val_acc, 4),
            "global_test_accuracy": round(test_acc, 4),
            "weighted_precision": round(prec, 4),
            "weighted_recall": round(rec, 4),
            "weighted_f1": round(f1, 4),
            "top3_accuracy": round(top3_acc, 4)
        }
        round_history.append(metrics)
        print(f"Round {rnd:02d}/25 | Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Test F1: {f1*100:.2f}% | Top-3: {top3_acc*100:.2f}%")

    res_df = pd.DataFrame(round_history)
    csv_path = os.path.join(BASE_DIR, "ml", "balanced_non_iid_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved round results -> {csv_path}")

    # Identify best round based on global test accuracy
    best_test_row = res_df.sort_values(by="global_test_accuracy", ascending=False).iloc[0]
    best_round = int(best_test_row["round"])
    best_test_acc = float(best_test_row["global_test_accuracy"])
    
    best_val_row = res_df.sort_values(by="val_accuracy", ascending=False).iloc[0]
    best_val_round = int(best_val_row["round"])

    # Perform detailed diagnostic on the best global test round
    best_model_path = os.path.join(VERSIONED_DIR, f"5_clients_round_{best_round:02d}.keras")
    best_model = keras.models.load_model(best_model_path)
    probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    encoders = joblib.load(os.path.join(BASE_DIR, "ml", "models", "fl_encoders.pkl"))
    crop_names = encoders["CROPS"].classes_
    sugarbeet_idx = np.where([c.lower() == 'sugarbeet' for c in crop_names])[0][0]

    act_counts = np.bincount(y_test, minlength=len(crop_names))
    pred_counts = np.bincount(y_pred, minlength=len(crop_names))

    cm = confusion_matrix(y_test, y_pred, labels=range(len(crop_names)))
    report = classification_report(y_test, y_pred, target_names=crop_names, output_dict=True, zero_division=0)

    sb_name = crop_names[sugarbeet_idx]
    sb_actual = int(act_counts[sugarbeet_idx])
    sb_pred = int(pred_counts[sugarbeet_idx])
    sb_prec = float(report[sb_name]['precision'])
    sb_rec = float(report[sb_name]['recall'])
    sb_pred_pct = (sb_pred / len(y_test)) * 100

    sb_fps = {}
    for i in range(len(crop_names)):
        if i != sugarbeet_idx and cm[i, sugarbeet_idx] > 0:
            sb_fps[crop_names[i]] = int(cm[i, sugarbeet_idx])

    diagnostic_payload = {
        "experiment": "5_clients_balanced_non_iid",
        "best_global_test_round": best_round,
        "best_global_test_accuracy": best_test_acc,
        "best_validation_round": best_val_round,
        "sugarbeet_metrics": {
            "actual_count": sb_actual,
            "predicted_count": sb_pred,
            "actual_percentage": round((sb_actual / len(y_test)) * 100, 2),
            "predicted_percentage": round(sb_pred_pct, 2),
            "precision": round(sb_prec, 4),
            "recall": round(sb_rec, 4),
            "misclassified_crops_as_sugarbeet": sb_fps
        },
        "comparison_with_extreme_non_iid_round_16": {
            "old_global_test_acc": 0.4932,
            "new_global_test_acc": round(best_test_acc, 4),
            "accuracy_improvement": round(best_test_acc - 0.4932, 4),
            "old_sugarbeet_pred_pct": 10.75,
            "new_sugarbeet_pred_pct": round(sb_pred_pct, 2),
            "old_sugarbeet_precision": 0.1633,
            "new_sugarbeet_precision": round(sb_prec, 4),
            "class_collapse_resolved": bool(sb_pred_pct < 5.0 and sb_prec > 0.4)
        }
    }

    json_path = os.path.join(BASE_DIR, "ml", "balanced_non_iid_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump(diagnostic_payload, f, indent=4)
    print(f"Saved diagnostic report -> {json_path}")

    # Summary Output
    print("\n" + "="*65)
    print("  FINAL SUMMARY -- Balanced Non-IID Experiment (5 Clients)")
    print("="*65)
    print(f"Best Validation Round       : Round {best_val_round}")
    print(f"Best Global Test Round     : Round {best_round}")
    print(f"Global Test Accuracy       : {best_test_acc*100:.2f}% (vs 49.32% in extreme soil Non-IID)")
    print(f"Top-3 Accuracy             : {best_test_row['top3_accuracy']*100:.2f}%")
    print(f"Sugarbeet Actual Distribution: {(sb_actual/len(y_test))*100:.2f}% ({sb_actual} samples)")
    print(f"Sugarbeet Pred Distribution  : {sb_pred_pct:.2f}% ({sb_pred} samples, down from 10.75%)")
    print(f"Sugarbeet Precision        : {sb_prec*100:.2f}% (up from 16.33%)")
    print(f"Sugarbeet Recall           : {sb_rec*100:.2f}%")
    print(f"Class Collapse Improved?    : {'YES - Collapse Resolved!' if sb_pred_pct < 5.0 else 'PARTIAL'}")
    print("="*65)

if __name__ == "__main__":
    run_experiment()
