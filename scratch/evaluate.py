import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

def main():
    base_dir = r"D:\Krishi"
    data_path = os.path.join(base_dir, "data", "raw", "primary.csv")
    models_dir = os.path.join(base_dir, "ml", "models")
    
    # 1. Load data
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    df = df.dropna()
    
    leakage_cols = [
        'TYPE_OF_CROP', 'HARVESTED',
        'SOIL_PH_HIGH', 'CROPDURATION_MAX', 'MAX_TEMP',
        'WATERREQUIRED_MAX', 'RELATIVE_HUMIDITY_MAX',
        'N_MAX', 'P_MAX', 'K_MAX',
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    # 2. Split (must match dnn_baseline.py and federated split)
    # The models were trained using random_state=42 stratify=y
    # We will use encoders to stratify
    encoders = joblib.load(os.path.join(models_dir, 'fl_encoders.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'fl_scaler.pkl'))
    
    # Encode for stratify
    df['CROPS'] = encoders['CROPS'].transform(df['CROPS'].astype(str))
    
    X_raw = df.drop(columns=['CROPS'])
    y = df['CROPS'].values
    
    _, X_test_raw, _, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply encoders exactly as ModelService does
    X_test_encoded = X_test_raw.copy()
    for col in ['SOIL', 'SOWN', 'WATER_SOURCE', 'SEASON']:
        if col in X_test_encoded.columns:
            X_test_encoded[col] = encoders[col].transform(X_test_encoded[col].astype(str))
            
    # Production feature order
    feature_order = ['SOIL', 'SEASON', 'SOWN', 'WATER_SOURCE', 'SOIL_PH', 'CROPDURATION', 
                     'TEMP', 'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']
    
    X_test_ordered = X_test_encoded[feature_order].values.astype(np.float32)
    X_test_scaled = scaler.transform(X_test_ordered)
    
    # 3. Load Models
    current_model_path = os.path.join(models_dir, 'fl_global_model.keras')
    best_checkpoint_path = os.path.join(models_dir, 'versioned', '5_clients_round_16.keras')
    
    current_model = keras.models.load_model(current_model_path)
    best_model = keras.models.load_model(best_checkpoint_path)
    
    def evaluate(model, name):
        probs = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        
        # Top-3 Accuracy
        top3_preds = np.argsort(probs, axis=1)[:, -3:]
        top3_acc = np.mean([1 if y_test[i] in top3_preds[i] else 0 for i in range(len(y_test))])
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Top-3 Acc: {top3_acc:.4f}")
        print()
        
    evaluate(current_model, "Current fl_global_model.keras")
    evaluate(best_model, "Best Checkpoint (5_clients_round_16.keras)")

if __name__ == "__main__":
    main()
