import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######

# Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    # Merge the dataframes on ndc_code
    # The ndc_df should have columns like 'NDC_Code' and 'Non-proprietary Name' (generic name)
    
    # First, let's see what columns are in ndc_df and standardize
    df = df.copy()
    
    # Create a mapping from NDC code to generic drug name
    # Handle different possible column names in the lookup table
    if 'Non-proprietary Name' in ndc_df.columns:
        generic_col = 'Non-proprietary Name'
    elif 'Nonproprietary Name' in ndc_df.columns:
        generic_col = 'Nonproprietary Name'
    elif 'generic_name' in ndc_df.columns:
        generic_col = 'generic_name'
    else:
        # Use first column that looks like a name
        generic_col = [c for c in ndc_df.columns if 'name' in c.lower()][0] if any('name' in c.lower() for c in ndc_df.columns) else ndc_df.columns[1]
    
    # Get the NDC code column
    if 'NDC_Code' in ndc_df.columns:
        ndc_col = 'NDC_Code'
    elif 'ndc_code' in ndc_df.columns:
        ndc_col = 'ndc_code'
    else:
        ndc_col = ndc_df.columns[0]
    
    # Create mapping dictionary
    ndc_mapping = dict(zip(ndc_df[ndc_col].astype(str), ndc_df[generic_col].astype(str)))
    
    # Map the ndc_code to generic_drug_name
    df['generic_drug_name'] = df['ndc_code'].astype(str).map(ndc_mapping)
    
    # Fill unmapped values with 'Other'
    df['generic_drug_name'] = df['generic_drug_name'].fillna('Other')
    
    return df


# Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
                              (keeping all drug lines for that encounter)
    '''
    # Sort by encounter_id (assuming lower = earlier)
    df_sorted = df.sort_values('encounter_id')
    
    # Find the first (minimum) encounter_id for each patient
    first_encounter_ids = df_sorted.groupby('patient_nbr')['encounter_id'].min().reset_index()
    first_encounter_ids = first_encounter_ids.rename(columns={'encounter_id': 'first_encounter_id'})
    
    # Merge to get all rows for each patient's first encounter
    df_merged = df_sorted.merge(first_encounter_ids, on='patient_nbr')
    
    # Keep only rows where encounter_id matches the first encounter
    first_encounter_df = df_merged[df_merged['encounter_id'] == df_merged['first_encounter_id']].copy()
    
    # Drop the helper column
    first_encounter_df = first_encounter_df.drop(columns=['first_encounter_id'])
    
    return first_encounter_df


# Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    # Get unique patients
    unique_patients = df[patient_key].unique()
    
    # Shuffle patients
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    # Split 60/20/20
    n_patients = len(unique_patients)
    train_end = int(n_patients * 0.6)
    val_end = int(n_patients * 0.8)
    
    train_patients = set(unique_patients[:train_end])
    val_patients = set(unique_patients[train_end:val_end])
    test_patients = set(unique_patients[val_end:])
    
    # Split dataframes
    train = df[df[patient_key].isin(train_patients)].copy()
    validation = df[df[patient_key].isin(val_patients)].copy()
    test = df[df[patient_key].isin(test_patients)].copy()
    
    return train, validation, test


# Question 7
def create_tf_categorical_feature_cols(categorical_col_list, vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir, c + "_vocab.txt")
        
        # Create categorical column from vocabulary file
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c,
            vocabulary_file=vocab_file_path,
            num_oov_buckets=1  # Out-of-vocabulary bucket for unseen values
        )
        
        # Wrap in indicator column (one-hot encoding)
        # For high cardinality features, could use embedding_column instead
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
        
        output_tf_list.append(tf_categorical_feature_column)
    
    return output_tf_list


# Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean) / std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    # Create normalizer function
    # Note: Handles std=0 (constant columns) and type mismatches to prevent NaN/Inf
    def zscore_normalizer(col_value):
        # Cast to float32 to fix int64/float32 type mismatch from dataset
        col_value = tf.cast(col_value, tf.float32)
        std_tensor = tf.constant(STD, dtype=tf.float32)
        mean_tensor = tf.constant(MEAN, dtype=tf.float32)
        # Use safe_std to prevent division by zero for constant columns (std=0)
        safe_std = tf.maximum(std_tensor, 1e-6)
        return (col_value - mean_tensor) / safe_std
    
    # Create numeric column with normalization
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col,
        default_value=default_value,
        normalizer_fn=zscore_normalizer
    )
    
    return tf_numeric_feature


# Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: Model prediction - either TF Probability distribution or tensor
    return:
        m: mean predictions
        s: standard deviation predictions
    '''
    import tensorflow as tf
    # Check if it's a TF Probability distribution (has mean() method)
    if hasattr(diabetes_yhat, 'mean') and callable(getattr(diabetes_yhat, 'mean')):
        m = diabetes_yhat.mean()
        s = diabetes_yhat.stddev()
    else:
        # Model outputs (batch, 2) where [:, 0] is mean and [:, 1] is std_raw
        # Convert std_raw to std: scale = 1e-3 + softplus(0.01 * std_raw)
        m = diabetes_yhat[:, 0]
        std_raw = diabetes_yhat[:, 1]
        s = 1e-3 + tf.nn.softplus(0.01 * std_raw)
    return m, s


# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str, probability mean prediction field
    return:
        student_binary_prediction: numpy array with binary labels (1 if >= 5 days, 0 otherwise)
    '''
    # Convert prediction to binary: 1 if predicted hospitalization >= 5 days
    student_binary_prediction = (df[col].values >= 5).astype(int)
    return student_binary_prediction
