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
├── student_project_submission.py     # Python script version
├── student_project_udacity.ipynb     # Udacity project exercises (TensorFlow NN)
├── student_project_udacity.py        # Python script version
├── utils.py                          # Utility functions
├── student_utils.py                  # Student implementation functions
└── requirements.txt                  # Python dependencies
```

## Files Description

### My Implementation (Original Dataset)

- **student_project_submission.ipynb**: My own implementation using the UCI Diabetes dataset (`diabetic_data.csv`). Uses Gradient Boosting Regressor after neural network attempts failed.
- **student_project_submission.py**: Python script version of the notebook for command-line execution.

### Udacity Project (Official Dataset)

- **student_project_udacity.ipynb**: Implementation following Udacity's project structure using `final_project_dataset.csv`. Uses TensorFlow neural network with feature columns.
- **student_project_udacity.py**: Python script version for testing the pipeline.

### Supporting Files

- **utils.py**: Helper functions for data processing, aggregation, and TensorFlow utilities. Includes `DenseFeaturesCompat` for TensorFlow 2.20+ compatibility.
- **student_utils.py**: Core functions for NDC reduction, encounter selection, patient splitting, and feature engineering.

## Model Comparison Results

Both models were trained and evaluated on the Udacity dataset (`final_project_dataset.csv`) for fair comparison:

| Metric | Gradient Boosting | TensorFlow NN | Winner |
|--------|------------------|---------------|--------|
| **MAE (days)** | 1.7130 | 1.7262 | Gradient Boosting |
| **RMSE (days)** | 2.2856 | 2.3118 | Gradient Boosting |
| **R² Score** | 0.4035 | 0.3897 | Gradient Boosting |
| **Accuracy** | 0.7631 | 0.7591 | Gradient Boosting |
| **F1-Score** | 0.6467 | 0.6216 | Gradient Boosting |
| **AUC** | 0.8203 | 0.8157 | Gradient Boosting |

**Result**: Gradient Boosting wins 6 out of 8 metrics. It provides better regression and classification performance, faster training, and more interpretable results through feature importance.

## Technical Notes

### DenseFeatures Deprecation

TensorFlow 2.20+ removed `tf.keras.layers.DenseFeatures` as part of deprecating the feature column API. I created `DenseFeaturesCompat` in `utils.py` as a drop-in replacement that uses the internal `input_layer` function. This maintains compatibility with existing feature column code while working with modern TensorFlow versions.

### TensorFlow Probability Compatibility Issues

TensorFlow Probability 0.25.0 with TensorFlow 2.20.0 has compatibility issues with `DenseVariational` and `DistributionLambda` layers when using the Functional API. These layers fail with `'tuple' object has no attribute 'rank'` errors due to KerasTensor handling in graph mode.

**Workaround**: The model uses `Dense(2)` to output mean and std_raw values directly, instead of using `DenseVariational` and `DistributionLambda`. The distribution can be created during prediction/evaluation: `Normal(loc=outputs[:,0], scale=1e-3+softplus(0.01*outputs[:,1]))`. This maintains the same model structure and probabilistic output while working around the compatibility issue.

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

3. Run scripts:
```bash
# My implementation (Gradient Boosting)
python3 student_project_submission.py

# Udacity project (TensorFlow NN)
python3 student_project_udacity.py
```

## Dataset Information

- **Original Dataset**: UCI Diabetes 130-US hospitals (1999-2008) - `diabetic_data.csv`
- **Udacity Dataset**: Modified synthetic dataset - `final_project_dataset.csv`
- Both datasets contain patient demographics, admission info, medical codes, procedures, and hospitalization time (target variable)

## References

- UCI Dataset: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
- Udacity Project: https://github.com/udacity/cd0372-Applying-AI-to-EHR-Data
- Data Schema: https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/project/data_schema_references
