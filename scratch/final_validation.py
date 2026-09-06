import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def evaluate_model(model, X, y):
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    
    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = np.mean([1 if y[i] in top3_preds[i] else 0 for i in range(len(y))])
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "top3_acc": top3_acc,
        "probs": probs,
        "y_pred": y_pred
    }

def main():
    base_dir = r"D:\Krishi"
    models_dir = os.path.join(base_dir, "ml", "models")
    old_model_path = os.path.join(models_dir, "fl_global_model.keras")
    new_model_path = os.path.join(models_dir, "balanced_non_iid", "versioned", "5_clients_round_24.keras")
    shards_dir = os.path.join(base_dir, "data", "processed", "client_shards", "balanced_non_iid", "5_clients")
    processed_dir = os.path.join(base_dir, "data", "processed")

    # Verification 1 & 2 & 4 & 9
    X_test_df = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test_df = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))
    X_train_df = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    
    test_ids = set(X_test_df["source_row_id"].astype(int))
    train_ids = set(X_train_df["source_row_id"].astype(int))
    
    shard_ids = set()
    for i in range(1, 6):
        shard_df = pd.read_csv(os.path.join(shards_dir, f"client_{i:02d}.csv"))
        shard_ids.update(shard_df["source_row_id"].astype(int))
    
    num_test_samples = len(X_test_df)
    num_train_samples = len(X_train_df)
    overlap = test_ids.intersection(shard_ids)
    
    pass_overlap = len(overlap) == 0
    pass_train_samples = num_train_samples == 45600 and len(shard_ids) == 45600
    pass_test_samples = num_test_samples == 11400

    # Model Evaluation setup
    feature_cols = ['SOIL', 'SEASON', 'SOWN', 'WATER_SOURCE', 'SOIL_PH', 'CROPDURATION', 
                    'TEMP', 'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']
    
    X_test = X_test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = y_test_df["target"].to_numpy(dtype=np.int32)
    
    encoders = joblib.load(os.path.join(models_dir, "fl_encoders.pkl"))
    num_classes = len(encoders['CROPS'].classes_)
    pass_num_classes = num_classes == 57

    old_model = keras.models.load_model(old_model_path)
    new_model = keras.models.load_model(new_model_path)

    # Verification 6, 7, 8
    def check_model_arch(m):
        in_shape = m.input_shape[1]
        out_shape = m.output_shape[1]
        is_softmax = 'softmax' in str(m.layers[-1].activation) if hasattr(m.layers[-1], 'activation') else False
        return in_shape == 12 and out_shape == 57 and is_softmax
        
    pass_old_arch = check_model_arch(old_model)
    pass_new_arch = check_model_arch(new_model)

    old_res = evaluate_model(old_model, X_test, y_test)
    new_res = evaluate_model(new_model, X_test, y_test)

    # Prob sum check
    old_prob_sums = np.sum(old_res["probs"], axis=1)
    new_prob_sums = np.sum(new_res["probs"], axis=1)
    pass_prob_sum = np.allclose(old_prob_sums, 1.0) and np.allclose(new_prob_sums, 1.0)

    # Verification 10
    results_csv = pd.read_csv(os.path.join(base_dir, "ml", "balanced_non_iid_results.csv"))
    best_round_df = results_csv.sort_values(by="global_test_accuracy", ascending=False).iloc[0]
    pass_best_round = int(best_round_df["round"]) == 24
    
    # 57 classes represented
    act_counts = np.bincount(y_test, minlength=57)
    pass_all_classes = np.all(act_counts > 0)
    
    print("\n=======================================================")
    print("      INDEPENDENT VALIDATION SUMMARY")
    print("=======================================================")
    print(f"Test Samples: {num_test_samples} (Expected: 11400) -> {'PASS' if pass_test_samples else 'FAIL'}")
    print(f"Train Samples: {len(shard_ids)} (Expected: 45600) -> {'PASS' if pass_train_samples else 'FAIL'}")
    print(f"Data Leakage: {len(overlap)} test IDs in train shards -> {'PASS' if pass_overlap else 'FAIL'}")
    print(f"Num Classes: {num_classes} -> {'PASS' if pass_num_classes and pass_all_classes else 'FAIL'}")
    print(f"Architecture: 12 In, 57 Out, Softmax -> {'PASS' if pass_new_arch else 'FAIL'}")
    print(f"Probabilities sum to 1.0 -> {'PASS' if pass_prob_sum else 'FAIL'}")
    print(f"Round 24 is highest performing -> {'PASS' if pass_best_round else 'FAIL'}")
    
    all_pass = all([pass_test_samples, pass_train_samples, pass_overlap, pass_num_classes, 
                    pass_all_classes, pass_new_arch, pass_prob_sum, pass_best_round])

    print(f"\nFINAL VALIDATION STATUS: {'PASS' if all_pass else 'FAIL'}")

    print("\n=======================================================")
    print("      METRICS COMPARISON")
    print("=======================================================")
    print(f"{'Metric':<20} | {'Old Model':<15} | {'New Round 24':<15} | {'Diff'}")
    print("-" * 65)
    metrics = [
        ("Accuracy", "accuracy"),
        ("Weighted Precision", "precision"),
        ("Weighted Recall", "recall"),
        ("Weighted F1", "f1"),
        ("Top-3 Accuracy", "top3_acc")
    ]
    for name, key in metrics:
        old_val = old_res[key] * 100
        new_val = new_res[key] * 100
        diff = new_val - old_val
        print(f"{name:<20} | {old_val:>14.2f}% | {new_val:>14.2f}% | {diff:>+6.2f}%")

if __name__ == "__main__":
    main()
