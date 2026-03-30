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

To run the automated pipeline, execute the main training script:

```bash
python train.py
```

The script will:

1. Download the Wisconsin Breast Cancer Dataset from UCI Machine Learning Repository.
2. Preprocess the data.
3. Train and evaluate all models.
4. Optimize hyperparameters using Optuna.
5. Save the best performing pipeline to `cancer_classification_pipeline.pkl`.
6. Log all results to `logs.log`.

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
