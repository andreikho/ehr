#!/usr/bin/env python3
"""
EHR Data Analysis: Hospitalization Time Prediction
Replicates student_project_submission.ipynb workflow using Gradient Boosting Regressor.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# Set random seeds for reproducibility
np.random.seed(42)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 80)

def load_and_prepare_data():
    """Load and prepare the diabetes dataset."""
    print_step(1, "Loading Dataset")
    data_path = 'diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv'
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of encounters: {len(df)}")
    print(f"Number of unique patients: {df['patient_nbr'].nunique()}")
    
    return df

def clean_data(df):
    """Clean and preprocess the dataset."""
    print_step(2, "Data Cleaning and Preprocessing")
    
    # Remove weight and payer_code (high missing values, fairness concerns)
    df = df.drop(columns=['weight', 'payer_code'], errors='ignore')
    
    # Handle '?' values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('?', np.nan)
    
    # Remove invalid gender entries
    df = df[df['gender'] != 'Unknown/Invalid']
    
    # Impute race and medical_specialty
    df['race'] = df['race'].fillna('Unknown')
    df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
    
    # Select first encounter per patient
    df = df.sort_values('encounter_id').groupby('patient_nbr').first().reset_index()
    
    print(f"After cleaning: {df.shape[0]:,} patients")
    
    return df

def prepare_features(df):
    """Prepare features for modeling."""
    print_step(3, "Feature Engineering")
    
    # Define feature columns
    TARGET_COL = 'time_in_hospital'
    
    # Categorical features (excluding diagnosis codes for simplicity)
    CATEGORICAL_COLS = [
        'race', 'gender', 'age', 'admission_type_id', 
        'discharge_disposition_id', 'admission_source_id',
        'max_glu_serum', 'A1Cresult', 'change', 'readmitted',
        'medical_specialty'
    ]
    
    # Medication columns (binary indicators)
    MEDICATION_COLS = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'diabetesMed'
    ]
    
    # Numerical features
    NUMERICAL_COLS = [
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses'
    ]
    
    # Combine all features
    feature_cols = CATEGORICAL_COLS + MEDICATION_COLS + NUMERICAL_COLS
    
    # Remove any columns that don't exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"Selected {len(feature_cols)} features")
    print(f"  Categorical: {len(CATEGORICAL_COLS)}")
    print(f"  Medications: {len(MEDICATION_COLS)}")
    print(f"  Numerical: {len(NUMERICAL_COLS)}")
    
    return df, feature_cols, TARGET_COL

def encode_features(df, feature_cols, categorical_cols):
    """Encode categorical features and normalize numerical features."""
    print_step(4, "Feature Encoding and Normalization")
    
    df_encoded = df[feature_cols + ['time_in_hospital']].copy()
    
    # Label encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].fillna('missing')
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    # Normalize all features (including encoded categorical)
    scaler = StandardScaler()
    feature_data = df_encoded[feature_cols]
    df_encoded[feature_cols] = scaler.fit_transform(feature_data)
    
    print("Features encoded and normalized")
    
    return df_encoded, label_encoders, scaler

def split_data(df_encoded, target_col):
    """Split data by patient to avoid data leakage."""
    print_step(5, "Splitting Dataset by Patient")
    
    from student_utils import patient_dataset_splitter
    
    # Use patient_nbr for splitting
    df_encoded['patient_nbr'] = df['patient_nbr'].values
    
    train_df, val_df, test_df = patient_dataset_splitter(df_encoded, 'patient_nbr')
    
    X_train = train_df.drop(columns=[target_col, 'patient_nbr']).values
    X_val = val_df.drop(columns=[target_col, 'patient_nbr']).values
    X_test = test_df.drop(columns=[target_col, 'patient_nbr']).values
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    print(f"Train: {len(X_train):,} patients")
    print(f"Validation: {len(X_val):,} patients")
    print(f"Test: {len(X_test):,} patients")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting Regressor."""
    print_step(6, "Training Gradient Boosting Regressor")
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"\nValidation Results:")
    print(f"  MAE: {val_mae:.4f} days")
    print(f"  RMSE: {val_rmse:.4f} days")
    print(f"  R²: {val_r2:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    print_step(7, "Evaluating Model on Test Set")
    
    # Regression metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Regression Metrics:")
    print(f"  MAE: {mae:.4f} days")
    print(f"  RMSE: {rmse:.4f} days")
    print(f"  R²: {r2:.4f}")
    
    # Binary classification metrics (threshold >= 5 days)
    binary_labels = (y_test >= 5.0).astype(int)
    binary_preds = (y_pred >= 5.0).astype(int)
    
    accuracy = accuracy_score(binary_labels, binary_preds)
    precision = precision_score(binary_labels, binary_preds, zero_division=0)
    recall = recall_score(binary_labels, binary_preds, zero_division=0)
    f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(binary_labels, y_pred)
    except:
        auc = 0.0
    
    print(f"\nBinary Classification Metrics (threshold >= 5 days):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def main():
    print_section("EHR Data Analysis: Hospitalization Time Prediction")
    
    # Load data
    df = load_and_prepare_data()
    
    # Clean data
    df = clean_data(df)
    
    # Prepare features
    df, feature_cols, target_col = prepare_features(df)
    
    # Get categorical columns
    categorical_cols = [col for col in feature_cols if col not in 
                       ['num_lab_procedures', 'num_procedures', 'num_medications',
                        'number_outpatient', 'number_emergency', 'number_inpatient',
                        'number_diagnoses']]
    
    # Encode features
    df_encoded, label_encoders, scaler = encode_features(df, feature_cols, categorical_cols)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded, target_col)
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Summary
    print_section("Summary")
    print("Model training and evaluation completed successfully!")
    print(f"\nFinal Test Results:")
    print(f"  MAE: {results['mae']:.4f} days")
    print(f"  RMSE: {results['rmse']:.4f} days")
    print(f"  R²: {results['r2']:.4f}")
    print(f"  Binary F1: {results['f1']:.4f}")
    print(f"  Binary AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
