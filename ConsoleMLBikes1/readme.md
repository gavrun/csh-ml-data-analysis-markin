# Bike Rental Type Prediction Project

## Overview

This project is a machine learning application built with ML.NET that predicts the type of bike rental (short-term or long-term) based on weather and time-related features. It demonstrates a full machine learning workflow: from loading data, preprocessing, training multiple models, choosing the best one, saving it, and making predictions.


## Project Structure

```
ConsoleMLBikes/
├── Program.cs
├── BikeRentalData.cs         # Data structure for input dataset
├── RentalTypePrediction.cs   # Data structure for model output
├── Data/
│   └── bike_sharing_2000.csv # Example dataset
└── Models/
    └── model.zip             # Saved trained model
```

- `BikeRentalData.cs`  
  Defines the data structure for loading the bike rental dataset. Each property represents a column from the CSV file (season, month, temperature, humidity, etc.).

- `RentalTypePrediction.cs`  
  Defines the structure of the model's prediction output, including the predicted label, probability, and score.

- `Program.cs`  
  Main program file that:
  - Loads the dataset.
  - Preprocesses data (encoding categorical features and normalizing numerical ones).
  - Trains and evaluates multiple machine learning models (Logistic Regression, FastTree, LightGBM).
  - Selects the best model based on F1 Score.
  - Saves the best model to a file.
  - Loads an existing model if available to avoid retraining.
  - Predicts rental type for a sample input.


## How It Works

1. **Load Dataset:**
   - Reads a CSV file containing historical bike rental data.

2. **Data Preprocessing:**
   - Categorical fields like season and weather condition are one-hot encoded.
   - Numerical fields like temperature, humidity, and windspeed are normalized.
   - All features are combined into a single feature vector.

3. **Model Training:**
   - Trains three different models:
     - Logistic Regression
     - FastTree (gradient boosting)
     - LightGBM (efficient gradient boosting)

4. **Model Evaluation:**
   - Evaluates each model using Accuracy, AUC, and F1 Score.
   - Selects the best model based on the highest F1 Score.

5. **Saving and Loading the Model:**
   - Saves the best model into the `Models/` directory as `model.zip`.
   - Automatically loads the existing model if available.

6. **Making Predictions:**
   - Predicts whether a bike rental is short-term or long-term for a given sample input.

