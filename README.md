# Automated Machine Learning Framework for Cancer Cell Classification

This project implements an automated machine learning (AutoML) framework for breast cancer cell classification using the Wisconsin Breast Cancer Dataset. The framework automates the entire machine learning pipeline, from data preprocessing and feature engineering to model training, evaluation, and hyperparameter optimization.

## Features

- **Automated Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features automatically.
- **Feature Engineering**: Generates new features to improve model performance.
- **Model Training**: Trains multiple machine learning models including:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- **Hyperparameter Optimization**: Uses Optuna to efficiently search for optimal hyperparameters for each model.
- **Performance Evaluation**: Evaluates models using comprehensive metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix
- **Pipeline Persistence**: Saves the best trained pipeline to a file for easy deployment.
- **Logging**: Comprehensive logging of all steps and results.

## Prerequisites

- Python 3.11
- pip

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd automated_ML_framework_for_cancer_cell_classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train and Save the Model

```bash
jupyter notebook train.ipynb
```

Run all notebook cells and ensure the final save step produces:

- `cancer_classification_pipeline.pkl`

### Run the Streamlit Interface

After the model file exists in the project root, start the interface with:

```bash
streamlit run app.py
```

The Streamlit app supports:

1. Single sample prediction via a form.
2. Batch prediction via CSV upload.
3. Downloading batch prediction results as CSV.

If `cancer_classification_pipeline.pkl` is missing, the app shows a guidance message and stops safely.

## Project Structure

```
automated_ML_framework_for_cancer_cell_classification/
├── cancer_classification_pipeline.pkl  # Saved trained pipeline
├── logs.log                            # Execution logs
├── python-installer.exe                # Python 3.11 installer
├── train.ipynb                         # Jupyter notebook with pipeline code
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Files to ignore in git
```

## License

This project is licensed under the terms of the MIT license.
