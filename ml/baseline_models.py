import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'raw', 'primary.csv')
    models_dir = os.path.join(base_dir, 'ml', 'models')
    reports_dir = os.path.join(base_dir, 'reports', 'metrics')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # 2. Exploratory checks
    print("\n--- Exploratory Checks ---")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nData Types:")
    print(df.dtypes)
    
    # 3. Remove duplicate rows
    print("\nRemoving duplicate rows...")
    df = df.drop_duplicates()
    
    # 4. Handle missing values
    if df.isnull().sum().sum() > 0:
        print("Handling missing values (dropping rows with NaNs)...")
        df = df.dropna()
        
    # 5. Drop columns causing data leakage.
    # TYPE_OF_CROP and HARVESTED are direct semantic leaks.
    # The _MAX columns (SOIL_PH_HIGH, CROPDURATION_MAX, MAX_TEMP, WATERREQUIRED_MAX,
    # RELATIVE_HUMIDITY_MAX, N_MAX, P_MAX, K_MAX) are crop requirement thresholds baked
    # into the dataset — each has exactly 1 unique value per crop, so the model was
    # simply reading the answer instead of learning from field measurements.
    columns_to_drop = [
        'TYPE_OF_CROP',
        'HARVESTED',
        'SOIL_PH_HIGH',
        'CROPDURATION_MAX',
        'MAX_TEMP',
        'WATERREQUIRED_MAX',
        'RELATIVE_HUMIDITY_MAX',
        'N_MAX',
        'P_MAX',
        'K_MAX',
    ]
    print(f"\nDropping {len(columns_to_drop)} leakage columns: {columns_to_drop}")
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # 6. Encode categorical columns
    categorical_cols = ['SOIL', 'SOWN', 'WATER_SOURCE', 'SEASON']
    encoders = {}
    
    print("\nEncoding categorical features...")
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    # 7. Encode target column
    print("Encoding target column (CROPS)...")
    target_le = LabelEncoder()
    df['CROPS'] = target_le.fit_transform(df['CROPS'].astype(str))
    encoders['CROPS'] = target_le
    
    # 8. Create X and y
    print("\nCreating feature matrix X and target vector y...")
    X = df.drop(columns=['CROPS'])
    y = df['CROPS']
    
    # 9. Train/Test Split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 10. Train RandomForestClassifier
    print("\nTraining RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 11. Evaluate Model
    print("\nEvaluating model...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_, zero_division=0))
    
    # 12. Feature Importance
    print("\nFeature Importance:")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    for i in range(X.shape[1]):
        print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")
        
    # 13. Plots
    print("\nGenerating plots...")
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'feature_importance.png'))
    plt.close()
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', 
                xticklabels=target_le.classes_, yticklabels=target_le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 14 & 15. Save models and encoders
    print(f"\nSaving model to {models_dir}\\random_forest_model.pkl")
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_model.pkl'))
    
    print(f"Saving encoders to {models_dir}\\encoders.pkl")
    joblib.dump(encoders, os.path.join(models_dir, 'encoders.pkl'))
    
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
