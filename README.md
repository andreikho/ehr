# EHR Data Analysis: Hospitalization Time Prediction Project

This project builds a regression model to predict expected hospitalization time for diabetes patients, which will be used to identify suitable candidates for a clinical trial requiring 5-7 days of hospital stay.

## Project Structure

```
ehr/
├── diabetes+130-us+hospitals+for+years+1999-2008/
│   ├── diabetic_data.csv          # Main dataset
│   └── IDS_mapping.csv            # Field mappings and descriptions
├── student_project_submission.ipynb  # Main Jupyter notebook
├── utils.py                       # Utility functions for data processing
├── student_utils.py               # Student implementation file
└── requirements.txt               # Python dependencies (pip)
```

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

3. Start Jupyter notebook:
```bash
jupyter notebook
```

## Project Requirements

The project requires:
- Python 3.8+
- TensorFlow 2.0+
- TensorFlow Probability
- TensorFlow Data Validation (optional)
- Aequitas for bias analysis
- Standard data science libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)

## Project Workflow

1. **Exploratory Data Analysis**: Analyze missing values, high cardinality fields, and distributions
2. **Data Preparation**: Select appropriate data level, map NDC codes, reduce dimensionality
3. **Feature Engineering**: Create categorical and numerical features using TensorFlow Feature Columns
4. **Model Building**: Build a deep learning regression model with TensorFlow Probability
5. **Model Evaluation**: Evaluate performance and convert to binary classification
6. **Bias Analysis**: Use Aequitas toolkit to analyze model biases across demographic groups

## Key Files

- **student_project_submission.ipynb**: Main notebook with project structure and TODOs
- **utils.py**: Helper functions for data loading, splitting, and TensorFlow dataset creation
- **student_utils.py**: Template file where students implement their feature engineering functions

## Dataset

The dataset is based on the UCI Diabetes readmission dataset (modified for this course). It contains:
- Patient demographics (race, gender, age)
- Admission information (type, source, disposition)
- Medical codes (ICD diagnosis codes, NDC drug codes)
- Lab procedures and medications
- Hospitalization time (target variable)

## References

- Dataset: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
- Data Schema: https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/project/data_schema_references
- Environment Setup: https://github.com/udacity/cd0372-Applying-AI-to-EHR-Data

