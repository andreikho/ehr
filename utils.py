"""
Utility functions for EHR data processing and TensorFlow dataset creation.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the diabetic data CSV file.
    
    Args:
        data_path: Path to the diabetic_data.csv file
        
    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(data_path)


def load_mapping(mapping_path: str) -> pd.DataFrame:
    """
    Load the IDS mapping CSV file.
    
    Args:
        mapping_path: Path to the IDS_mapping.csv file
        
    Returns:
        DataFrame containing the mapping data
    """
    return pd.read_csv(mapping_path)


def z_score_normalizer(mean: float, std: float) -> callable:
    """
    Create a z-score normalizer function.
    
    Args:
        mean: Mean value for normalization
        std: Standard deviation for normalization
        
    Returns:
        Normalizer function that takes a value and returns normalized value
    """
    def normalize(value):
        if std == 0:
            return 0.0
        return (value - mean) / std
    
    return normalize


def split_dataset_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets by patient_nbr.
    This ensures no data leakage between splits.
    
    Args:
        df: DataFrame with patient_nbr column
        train_ratio: Proportion of patients for training
        val_ratio: Proportion of patients for validation
        test_ratio: Proportion of patients for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Get unique patients
    unique_patients = df['patient_nbr'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_patients)
    
    # Calculate split indices
    n_patients = len(unique_patients)
    train_end = int(n_patients * train_ratio)
    val_end = train_end + int(n_patients * val_ratio)
    
    # Split patients
    train_patients = unique_patients[:train_end]
    val_patients = unique_patients[train_end:val_end]
    test_patients = unique_patients[val_end:]
    
    # Split dataframe by patient
    train_df = df[df['patient_nbr'].isin(train_patients)].copy()
    val_df = df[df['patient_nbr'].isin(val_patients)].copy()
    test_df = df[df['patient_nbr'].isin(test_patients)].copy()
    
    return train_df, val_df, test_df


def create_tf_dataset(
    df: pd.DataFrame,
    feature_columns: List[tf.feature_column.FeatureColumn],
    label_column: str,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from a pandas DataFrame.
    
    Args:
        df: Input DataFrame
        feature_columns: List of TensorFlow feature columns
        label_column: Name of the label column
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling
        
    Returns:
        TensorFlow Dataset
    """
    # Convert DataFrame to dictionary of tensors
    features_dict = {}
    for col in df.columns:
        if col != label_column:
            # Convert to numpy array and ensure proper dtype
            features_dict[col] = df[col].values
    
    # Create dataset from dictionary
    dataset = tf.data.Dataset.from_tensor_slices((features_dict, df[label_column].values))
    
    # Apply feature columns transformation
    def map_features(features, label):
        # Convert features dict to proper format for feature columns
        dense_tensor = tf.feature_column.input_layer(features, feature_columns)
        return dense_tensor, label
    
    dataset = dataset.map(map_features)
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    return dataset


def check_encounter_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> bool:
    """
    Check if there's any encounter or patient leakage between splits.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        
    Returns:
        True if no leakage detected, False otherwise
    """
    train_encounters = set(train_df['encounter_id'].unique())
    train_patients = set(train_df['patient_nbr'].unique())
    
    val_encounters = set(val_df['encounter_id'].unique())
    val_patients = set(val_df['patient_nbr'].unique())
    
    test_encounters = set(test_df['encounter_id'].unique())
    test_patients = set(test_df['patient_nbr'].unique())
    
    # Check for encounter leakage
    encounter_leakage = (
        len(train_encounters & val_encounters) > 0 or
        len(train_encounters & test_encounters) > 0 or
        len(val_encounters & test_encounters) > 0
    )
    
    # Check for patient leakage
    patient_leakage = (
        len(train_patients & val_patients) > 0 or
        len(train_patients & test_patients) > 0 or
        len(val_patients & test_patients) > 0
    )
    
    if encounter_leakage:
        print("WARNING: Encounter leakage detected!")
        return False
    
    if patient_leakage:
        print("WARNING: Patient leakage detected!")
        return False
    
    print("✓ No data leakage detected between splits.")
    return True

