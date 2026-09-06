import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

def main():
    base_dir = r"D:\Krishi"
    data_path = os.path.join(base_dir, "data", "raw", "primary.csv")
    models_dir = os.path.join(base_dir, "ml", "models")
    
    # Load data
    df = pd.read_csv(data_path).drop_duplicates().dropna()
    leakage_cols = [
        'TYPE_OF_CROP', 'HARVESTED', 'SOIL_PH_HIGH', 'CROPDURATION_MAX', 'MAX_TEMP',
        'WATERREQUIRED_MAX', 'RELATIVE_HUMIDITY_MAX', 'N_MAX', 'P_MAX', 'K_MAX',
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    encoders = joblib.load(os.path.join(models_dir, 'fl_encoders.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'fl_scaler.pkl'))
    
    df['CROPS'] = encoders['CROPS'].transform(df['CROPS'].astype(str))
    X_raw = df.drop(columns=['CROPS'])
    y = df['CROPS'].values
    
    _, X_test_raw, _, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
    
    X_test_encoded = X_test_raw.copy()
    for col in ['SOIL', 'SOWN', 'WATER_SOURCE', 'SEASON']:
        if col in X_test_encoded.columns:
            X_test_encoded[col] = encoders[col].transform(X_test_encoded[col].astype(str))
            
    feature_order = ['SOIL', 'SEASON', 'SOWN', 'WATER_SOURCE', 'SOIL_PH', 'CROPDURATION', 
                     'TEMP', 'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']
    
    X_test_ordered = X_test_encoded[feature_order].values.astype(np.float32)
    X_test_scaled = scaler.transform(X_test_ordered)
    
    # Load model
    current_model_path = os.path.join(models_dir, 'fl_global_model.keras')
    model = keras.models.load_model(current_model_path)
    
    # Model config
    in_shape = model.input_shape
    out_shape = model.output_shape
    
    # Predict
    probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    
    # Probability sums
    prob_sums = np.sum(probs, axis=1)
    min_sum, max_sum = np.min(prob_sums), np.max(prob_sums)
    is_softmax = 'softmax' in str(model.layers[-1].activation) if hasattr(model.layers[-1], 'activation') else False
    
    crop_names = encoders['CROPS'].classes_
    sugarbeet_idx = np.where(crop_names == 'sugarbeet')[0]
    
    if len(sugarbeet_idx) == 0:
        sugarbeet_idx = np.where(crop_names == 'Sugarbeet')[0]
        if len(sugarbeet_idx) == 0:
            sugarbeet_idx = np.where([c.lower() == 'sugarbeet' for c in crop_names])[0]
    
    sugarbeet_idx = sugarbeet_idx[0] if len(sugarbeet_idx) > 0 else -1
    
    # Confidence stats
    mean_conf = float(np.mean(max_probs))
    median_conf = float(np.median(max_probs))
    min_conf = float(np.min(max_probs))
    max_conf = float(np.max(max_probs))
    below_50 = float(np.mean(max_probs < 0.5) * 100)
    above_90 = float(np.mean(max_probs > 0.9) * 100)
    
    # Distributions
    act_counts = np.bincount(y_test, minlength=len(crop_names))
    pred_counts = np.bincount(y_pred, minlength=len(crop_names))
    
    dist_df = pd.DataFrame({
        'crop': crop_names,
        'actual_count': act_counts,
        'predicted_count': pred_counts,
        'actual_percentage': (act_counts / len(y_test)) * 100,
        'predicted_percentage': (pred_counts / len(y_pred)) * 100
    })
    
    cm = confusion_matrix(y_test, y_pred, labels=range(len(crop_names)))
    cm_df = pd.DataFrame(cm, index=crop_names, columns=crop_names)
    
    report = classification_report(y_test, y_pred, target_names=crop_names, output_dict=True, zero_division=0)
    
    # Sugarbeet details
    sugarbeet_name = crop_names[sugarbeet_idx] if sugarbeet_idx != -1 else "Unknown"
    sb_actual = int(act_counts[sugarbeet_idx]) if sugarbeet_idx != -1 else 0
    sb_pred = int(pred_counts[sugarbeet_idx]) if sugarbeet_idx != -1 else 0
    sb_prec = report.get(sugarbeet_name, {}).get('precision', 0)
    sb_rec = report.get(sugarbeet_name, {}).get('recall', 0)
    
    sb_fps = {}
    if sugarbeet_idx != -1:
        for i in range(len(crop_names)):
            if i != sugarbeet_idx and cm[i, sugarbeet_idx] > 0:
                sb_fps[crop_names[i]] = int(cm[i, sugarbeet_idx])
                
    # Root cause analysis hint
    imbalance = sb_actual / len(y_test) > 0.1
    collapse = sb_pred / len(y_test) > 0.1 and sb_actual / len(y_test) < 0.05
    
    root_cause = "B. Model bias/collapse (over-predicting Sugarbeet regardless of input, typical of non-IID client drift or straggler effects for this class)" if collapse else "F. Other/Unknown"
    
    # JSON structure
    out_json = {
        "model_architecture": {
            "input_shape": in_shape,
            "output_shape": out_shape,
            "is_softmax": is_softmax,
            "min_prob_sum": float(min_sum),
            "max_prob_sum": float(max_sum)
        },
        "target_mapping": {i: name for i, name in enumerate(crop_names)},
        "confidence_stats": {
            "mean": mean_conf,
            "median": median_conf,
            "min": min_conf,
            "max": max_conf,
            "percentage_below_50": below_50,
            "percentage_above_90": above_90
        },
        "sugarbeet_investigation": {
            "class_index": int(sugarbeet_idx),
            "crop_name": sugarbeet_name,
            "actual_samples": sb_actual,
            "predicted_samples": sb_pred,
            "precision": float(sb_prec),
            "recall": float(sb_rec),
            "false_positives_source_crops": sb_fps
        },
        "root_cause_diagnosis": root_cause
    }
    
    # Output to files
    dist_df.to_csv(os.path.join(base_dir, 'ml', 'sugarbeet_prediction_distribution.csv'), index=False)
    cm_df.to_csv(os.path.join(base_dir, 'ml', 'sugarbeet_confusion_matrix.csv'))
    with open(os.path.join(base_dir, 'ml', 'sugarbeet_bias_diagnostic.json'), 'w') as f:
        json.dump(out_json, f, indent=4)
        
    print("\n--- SUGARBEET BIAS DIAGNOSTIC ---")
    print(f"Model Input: {in_shape} | Output: {out_shape} | Softmax: {is_softmax}")
    print(f"Probability sums: min={min_sum:.6f}, max={max_sum:.6f}\n")
    
    print("--- Target Encoder ---")
    print(f"Total classes: {len(crop_names)}")
    print(f"Sugarbeet Index: {sugarbeet_idx} (Name: {sugarbeet_name})\n")
    
    print("--- Confidence Stats ---")
    print(f"Mean: {mean_conf:.4f} | Median: {median_conf:.4f}")
    print(f"Min: {min_conf:.4f} | Max: {max_conf:.4f}")
    print(f"< 50% Confidence: {below_50:.2f}%")
    print(f"> 90% Confidence: {above_90:.2f}%\n")
    
    print("--- Sugarbeet Focus ---")
    print(f"Actual Test Samples : {sb_actual} ({(sb_actual/len(y_test))*100:.2f}%)")
    print(f"Predicted Samples   : {sb_pred} ({(sb_pred/len(y_test))*100:.2f}%)")
    print(f"Precision           : {sb_prec:.4f}")
    print(f"Recall              : {sb_rec:.4f}")
    
    print(f"\nTop 5 Crops Incorrectly Predicted as Sugarbeet:")
    sorted_fps = sorted(sb_fps.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_fps[:5]:
        print(f"  {k}: {v} times")
        
    print(f"\nDiagnosis: {root_cause}")

if __name__ == "__main__":
    main()
