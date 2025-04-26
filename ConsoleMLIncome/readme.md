# Income Prediction Project

## Overview

This study project demonstrates a full ML workflow in ML.NET, from raw data to a ready-to-use, saved model for realistic predictions.

Real-world application here is to develop a machine learning model to predict whether a person's income exceeds $50K per year based on available demographic data.

## Project Tasks
- Load and analyze the dataset
- Preprocess numerical and categorical features
- Select and train models
- Evaluate model performance
- Visualize results
- Save and use the final model

## Dataset Description
- **Dataset:** Modified Adult Income Dataset
- **Source:** UCI Machine Learning Repository
- **Size:** ~48,000 instances

### Numerical Features
- `Age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`

### Categorical Features
- `Workclass`, `Education`, `MaritalStatus`, `Occupation`, `Relationship`, `Race`, `Sex`, `NativeCountry`

### Target Variable
- `Income`: `>50K` or `<=50K`

## Project Structure
```
ConsoleML.sln
└── ConsoleMLIncome
    ├─ Program.cs
    ├─ Data (adult.csv)
    ├─ DataProcessing (DataProcessor.cs)
    ├─ Evaluation (ModelEvaluator.cs)
    ├─ Models (AdultData.cs, AdultPrediction.cs, CustomMapping.cs, TrainTestData.cs)
    ├─ Prediction (PredictionEngine.cs)
    └─ Training (ModelTrainer.cs)
```

## Data Handling
- Load and inspect the dataset
- Calculate descriptive statistics for numerical features
- Analyze distribution of categorical features
- Analyze target variable distribution

## Data Preprocessing

### 1. Train/Test Split
- 80% training, 20% testing

### 2. Missing Values Handling
- Replace `?`, `NA`, `Unknown` with `null`
- Fill missing numeric fields

### 3. Feature Transformation
- Normalize numerical features
- Apply logarithmic transformation to `CapitalGain` and `CapitalLoss`
- One-Hot Encode categorical variables
- Hash encode high-cardinality categorical features
- Ordinal encoding for `Education` and `MaritalStatus`

### 4. Feature Engineering
- Create `HasCapitalIncome`, `AgeGroup`, and `WorkHoursCategory`

### 5. Feature Vector Creation
- Combine all transformed features into a single `Features` vector

## Model Training and Comparison

### Pipeline Application
- Apply preprocessing pipeline to the training set

### Training and Evaluation
- Combine preprocessing and algorithms
- Train models and evaluate:
  - Accuracy
  - AUC (Area Under ROC Curve)
  - F1 Score
  - Training Time

### Algorithms Compared
- SDCA Logistic Regression
- FastTree
- FastForest
- LightGBM
- Linear SVM

### Best Model Selection
- Based on highest AUC

## Hyperparameter Tuning and Final Training

### LightGBM Hyperparameter Search
- Grid search over combinations of `NumLeaves`, `NumTrees`, and `MinDataPerLeaf`
- 5-Fold Cross-Validation
- Select best combination by average AUC

### Final Training
- Train final model using best hyperparameters
- Evaluate on the test set
- Save the model to `income_prediction_model.zip`

## Visualization and Analysis

### Feature Importance
- Permutation Feature Importance to evaluate contributions to model performance

### Confusion Matrix
- Print counts of TP, FP, TN, FN
- Display matrix layout in console

### ROC Curve Preparation
- Setup for ROC visualization (optional)

## Model Usage in Console Application

### Interactive Mode
- Load the model
- Create `PredictionEngine`
- User inputs data via console prompts
- Model predicts income level
- Output includes prediction and probability

## Final Results
- Fully functional end-to-end machine learning pipeline
- Trained, optimized, and evaluated LightGBM model
- Ready-to-use model with user interaction support

## Steps to Run the Project

TBD.