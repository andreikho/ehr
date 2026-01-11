import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
from student_utils import create_tf_numeric_feature

def aggregate_dataset(df, grouping_field_list, array_field):
    """Aggregate dataset by grouping fields and create dummy columns for array field.
    
    Groups by encounter_id and aggregates drug names into dummy columns.
    Other fields take their first value (should be constant within encounter).
    """
    # Use encounter_id as the primary grouping key
    key_field = 'encounter_id'
    
    # Fields that should take first value (constant within encounter)
    other_fields = [c for c in grouping_field_list if c != key_field and c != 'ndc_code']
    
    # Build aggregation dict
    agg_dict = {array_field: list}  # Collect drugs as list
    for field in other_fields:
        agg_dict[field] = 'first'  # Take first value
    
    # Group by encounter_id and aggregate
    agg_df = df.groupby(key_field, as_index=False).agg(agg_dict)
    agg_df = agg_df.rename(columns={array_field: array_field + "_array"})
    
    # Create dummy columns from the array field (one-hot encoding of drugs)
    drug_series = agg_df[array_field + '_array'].apply(pd.Series).stack()
    if len(drug_series) > 0:
        dummy_df = pd.get_dummies(drug_series).groupby(level=0).sum()
        # Clean column names (replace spaces with underscores)
        dummy_col_list = [str(x).replace(" ", "_") for x in list(dummy_df.columns)]
        dummy_df.columns = dummy_col_list
    else:
        dummy_df = pd.DataFrame()
        dummy_col_list = []
    
    # Concatenate the aggregated dataframe with dummy columns
    concat_df = pd.concat([agg_df.reset_index(drop=True), dummy_df.reset_index(drop=True)], axis=1)
    
    # Clean all column names
    new_col_list = [str(x).replace(" ", "_") for x in list(concat_df.columns)]
    concat_df.columns = new_col_list

    return concat_df, dummy_col_list

def cast_df(df, col, d_type=str):
    return df[col].astype(d_type)

def impute_df(df, col, impute_value=0):
    return df[col].fillna(impute_value)
    
def preprocess_df(df, categorical_col_list, numerical_col_list, predictor, categorical_impute_value='nan',             numerical_impute_value=0):
    df = df.copy()  # Avoid SettingWithCopyWarning by working with a copy
    df[predictor] = df[predictor].astype(float)
    for c in categorical_col_list:
        df[c] = cast_df(df, c, d_type=str)
    for numerical_column in numerical_col_list:
        df[numerical_column] = impute_df(df, numerical_column, numerical_impute_value)
    return df

#adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, predictor,  batch_size=32):
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

# build vocab for categorical features
def write_vocabulary_file(vocab_list, field_name, default_value, vocab_dir='./diabetes_vocab/'):
    # Create directory if it doesn't exist
    os.makedirs(vocab_dir, exist_ok=True)
    output_file_path = os.path.join(vocab_dir, str(field_name) + "_vocab.txt")
    # put default value in first row as TF requires
    vocab_list = np.insert(vocab_list, 0, default_value, axis=0) 
    df = pd.DataFrame(vocab_list).to_csv(output_file_path, index=None, header=None)
    return output_file_path

def build_vocab_files(df, categorical_column_list, default_value='00'):
    vocab_files_list = []
    for c in categorical_column_list:
        v_file = write_vocabulary_file(df[c].unique(), c, default_value)
        vocab_files_list.append(v_file)
    return vocab_files_list

def show_group_stats_viz(df, group):
    print(df.groupby(group).size())
    print(df.groupby(group).size().plot(kind='barh'))
 
'''
Adapted from Tensorflow Probability Regression tutorial  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb    
'''
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2*n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t[..., :n],
                                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

def demo(feature_column, example_batch):
    """
    Demo function to test a feature column.
    Uses DenseFeaturesCompat for TensorFlow 2.20+ compatibility.
    """
    feature_layer = DenseFeaturesCompat([feature_column])
    print(feature_layer(example_batch))
    return feature_layer(example_batch)

def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std

def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list


# ============================================================================
# DenseFeatures Compatibility Layer for TensorFlow 2.20+
# ============================================================================

class DenseFeaturesCompat(tf.keras.layers.Layer):
    """
    Compatibility layer to replace tf.keras.layers.DenseFeatures (removed in TF 2.20+).
    
    This layer transforms TensorFlow feature columns into dense tensors, maintaining
    compatibility with the existing feature column API from the Udacity course.
    
    Why this alternative?
    ---------------------
    1. **API Compatibility**: Maintains compatibility with existing feature column code,
       avoiding a complete rewrite of feature engineering logic that uses vocabulary files
       and the feature column API.
    
    2. **Direct Feature Column Transformation**: Uses TensorFlow's feature column
       transformation API directly, which still works even though DenseFeatures wrapper
       was removed. This preserves the exact same transformations (normalization,
       vocabulary lookup, one-hot encoding) as the original DenseFeatures.
    
    3. **Seamless Integration**: Drop-in replacement for DenseFeatures - existing model
       architectures don't need to change, just replace:
       `tf.keras.layers.DenseFeatures(feature_columns)` 
       with 
       `DenseFeaturesCompat(feature_columns)`
    
    4. **Future-Proof**: While TensorFlow recommends migrating to Keras preprocessing layers
       long-term, this compatibility layer allows the course code to work immediately with
       modern TensorFlow versions while maintaining the same feature engineering approach.
    
    Technical Approach:
    ------------------
    The layer uses the internal input_layer function from tensorflow.python.feature_column
    which is the underlying function that DenseFeatures used. While the public API was
    removed in TF 2.20+, the internal implementation still exists and works. We wrap it
    in a Keras layer to maintain the same API as the original DenseFeatures.
    
    Note: This uses an internal TensorFlow API which may change in future versions.
    For production code, consider migrating to Keras preprocessing layers long-term.
    
    Usage:
        # Old (TF < 2.20):
        # feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        
        # New (TF 2.20+):
        feature_layer = DenseFeaturesCompat(feature_columns)
    """
    
    def __init__(self, feature_columns, name='dense_features_compat', **kwargs):
        super().__init__(name=name, **kwargs)
        self.feature_columns = feature_columns
        
    def call(self, inputs):
        """
        Transform feature columns to dense tensors.
        Uses the internal input_layer function which still exists in the feature_column_lib.
        This is the same underlying transformation that DenseFeatures used.
        """
        # Import the internal input_layer function
        # This still exists even though the public API was removed
        from tensorflow.python.feature_column import feature_column_lib as fc_lib
        return fc_lib.input_layer(inputs, self.feature_columns)