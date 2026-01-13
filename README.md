# EHR Data Analysis: Hospitalization Time Prediction Project

This project builds regression models to predict expected hospitalization time for diabetes patients, identifying suitable candidates for a clinical trial requiring 5-7 days of hospital stay.

## Project Overview

I started with the UCI Diabetes dataset (1999-2008) mentioned in the Udacity project description and built my own implementation. After trying a neural network that collapsed to mean predictions (4-4.5 days), I switched to Gradient Boosting which performed much better. Later, I found the official Udacity project repository with a different dataset and completed those exercises as well.

## Project Structure

```
ehr/
├── diabetes+130-us+hospitals+for+years+1999-2008/
│   ├── diabetic_data.csv          # Original UCI dataset
│   └── IDS_mapping.csv            # Field mappings
├── data/
│   └── final_project_dataset.csv  # Udacity project dataset
├── medication_lookup_tables/
│   └── final_ndc_lookup_table     # NDC code mappings
├── student_project_submission.ipynb  # My implementation (Gradient Boosting)
├── student_project_submission.html   # HTML export of submission notebook
├── student_project_udacity.ipynb     # Udacity project exercises (TensorFlow NN)
├── student_project_udacity.html      # HTML export of Udacity notebook
├── utils.py                          # Utility functions
├── student_utils.py                  # Student implementation functions
└── requirements.txt                  # Python dependencies
```

## Files Description

### My Implementation (Original Dataset)

- **student_project_submission.ipynb**: My own implementation using the UCI Diabetes dataset (`diabetic_data.csv`). Uses Gradient Boosting Regressor after neural network attempts failed.
- **student_project_submission.html**: HTML export of the submission notebook for easy viewing without running Jupyter.

### Udacity Project (Official Dataset)

- **student_project_udacity.ipynb**: Implementation following Udacity's project structure using `final_project_dataset.csv`. Uses TensorFlow neural network with feature columns.
- **student_project_udacity.html**: HTML export of the Udacity notebook for easy viewing without running Jupyter.

### Supporting Files

- **utils.py**: Helper functions for data processing, aggregation, and TensorFlow utilities. Includes `DenseFeaturesCompat` for TensorFlow 2.20+ compatibility.
- **student_utils.py**: Core functions for NDC reduction, encounter selection, patient splitting, and feature engineering.

## Model Comparison Results

**Important Note**: The two notebooks use **different datasets**, so this comparison reflects performance on different data distributions:

- **Gradient Boosting** (`student_project_submission.ipynb`): Trained on the original UCI Diabetes dataset (`diabetic_data.csv` - 101,767 rows)
- **TensorFlow NN** (`student_project_udacity.ipynb`): Trained on the Udacity modified dataset (`final_project_dataset.csv` - 143,425 rows)

The datasets have different sizes and potentially different distributions, so this comparison should be interpreted with that context in mind:

| Metric | Gradient Boosting | TensorFlow NN | Winner |
|--------|------------------|---------------|--------|
| **MAE (days)** | 1.636 | ~1.72 | Gradient Boosting |
| **RMSE (days)** | 2.194 | ~2.3 | Gradient Boosting |
| **R² Score** | 0.464 | ~0.39 | Gradient Boosting |
| **Accuracy** | ~0.77 | 0.5626 | Gradient Boosting |
| **F1-Score** | 0.670 | 0.3075 | Gradient Boosting |
| **AUC** | 0.836 | 0.5042 | Gradient Boosting |

**Result**: Gradient Boosting significantly outperforms the TensorFlow NN across all metrics. The TensorFlow model shows poor classification performance (AUC = 0.5042 indicates essentially random predictions), likely due to underfitting and model collapse to mean predictions. Gradient Boosting provides better regression and classification performance, faster training, and more interpretable results through feature importance.

**Note**: The TensorFlow NN's poor performance (especially AUC ≈ 0.5) suggests the model is not learning meaningful patterns and is essentially predicting at random for binary classification.

**Possible Contributing Factors**: The numerous compatibility issues encountered (see Issues Encountered section) required significant workarounds that may have impacted model performance:
- Using `Dense(2)` instead of `DenseVariational`/`DistributionLambda` changes the probabilistic learning approach
- Functional API workaround instead of Sequential API may affect gradient flow
- NaN prevention measures (Lambda layers, gradient clipping, lower learning rate) could limit the model's ability to learn complex patterns
- The model architecture had to be simplified to work around compatibility issues

While these workarounds were necessary for the code to run, they may have prevented the model from learning effectively. The poor results could be due to these constraints rather than fundamental limitations of neural networks for this task. Gradient Boosting, which didn't require such workarounds, was able to learn the patterns successfully.

## Technical Notes

### DenseFeatures Deprecation

TensorFlow 2.20+ removed `tf.keras.layers.DenseFeatures` as part of deprecating the feature column API. I created `DenseFeaturesCompat` in `utils.py` as a drop-in replacement that uses the internal `input_layer` function. This maintains compatibility with existing feature column code while working with modern TensorFlow versions.

### TensorFlow Probability Compatibility Issues

TensorFlow Probability 0.25.0 with TensorFlow 2.20.0 has compatibility issues with `DenseVariational` and `DistributionLambda` layers when using the Functional API. These layers fail with `'tuple' object has no attribute 'rank'` errors due to KerasTensor handling in graph mode.

**Workaround**: The model uses `Dense(2)` to output mean and std_raw values directly, instead of using `DenseVariational` and `DistributionLambda`. The distribution can be created during prediction/evaluation: `Normal(loc=outputs[:,0], scale=1e-3+softplus(0.01*outputs[:,1]))`. This maintains the same model structure and probabilistic output while working around the compatibility issue.

## Issues Encountered and Solutions

During implementation of the Udacity project, we encountered several compatibility and technical issues that required workarounds:

1. **DenseFeatures Removal**: TensorFlow 2.20+ removed `tf.keras.layers.DenseFeatures`, so we created `DenseFeaturesCompat` as a compatibility layer that uses the internal `input_layer` function.

2. **TFP Layer Compatibility**: TensorFlow Probability 0.25.0 + TF 2.20.0 has issues with `DenseVariational`/`DistributionLambda`, so we use `Dense(2)` to output mean and std_raw instead.

3. **Sequential API Limitation**: Sequential API doesn't work with TFP layers in newer TF versions, so we use Functional API while maintaining the same model structure.

4. **CPU-Only Training**: Added CPU-only configuration (`CUDA_VISIBLE_DEVICES=-1`) for users without GPU access.

5. **NaN Loss During Training**: Implemented NaN prevention measures including Lambda layer for NaN/Inf replacement, BatchNormalization, gradient clipping, and lower learning rate to stabilize training.

6. **Array Length Mismatch**: Fixed DataFrame creation error by using `preds[:, 0]` instead of `preds.flatten()` to match array lengths (preds is shape `(n_samples, 2)`).

7. **Type Mismatch in Normalization**: Added `tf.cast(col_value, tf.float32)` to fix int64/float32 type mismatch from dataset inputs.

8. **Division by Zero in Normalization**: Used `tf.maximum(std, 1e-6)` to prevent division by zero for constant columns (std=0) that would cause NaN values.

### Why Gradient Boosting?

Initially tried a TensorFlow neural network but it collapsed to predicting the mean (4-4.5 days) regardless of input. Gradient Boosting performed significantly better:
- Better handles tabular data with mixed feature types
- More robust to feature scaling issues
- Provides feature importance for interpretability
- Faster training and inference

## Setup Instructions

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run notebooks:
```bash
# Open Jupyter notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Dataset Information

- **Original Dataset**: UCI Diabetes 130-US hospitals (1999-2008) - `diabetic_data.csv`
- **Udacity Dataset**: Modified synthetic dataset - `final_project_dataset.csv`
- Both datasets contain patient demographics, admission info, medical codes, procedures, and hospitalization time (target variable)

## References

- UCI Dataset: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
- Udacity Project: https://github.com/udacity/cd0372-Applying-AI-to-EHR-Data
- Data Schema: https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/project/data_schema_references
