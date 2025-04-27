# Spam Detection Project 

## Overview
This is a study project aimed to demonstrate how to apply ML.NET Model Builder to build a machine learning model that classifies **email messages** as either **spam** or **not spam**.

The focus of the project is to understand the complete flow of machine learning in .NET applications using automated tools.

## Project Structure

```
ConsoleMLSpam
├── Dependencies
├── Data
│   └── email_dataset.csv          // CSV file with labeled email data
├── MLModel.mbconfig                // Model Builder configuration file
├── MLModel
│   ├── MLModel.consumption.cs      // Code to load the model and make predictions
│   ├── MLModel.ml.net              // Model definition and training settings (internal config)
│   └── MLModel.training.cs         // Code to retrain the model
└── Program.cs                      // Main console application entry point
```

## How It Works

1. **Create Project:** .NET Console App created in Visual Studio.
2. **Add Model Builder:** Integrated ML.NET Model Builder via "Add > Machine Learning".
3. **Load Data:** Loaded `email_dataset.csv`, mapped `Content` as input and `IsSpam` as label.
4. **Train Model:** Used **TextClassificationMulti** trainer; trained locally with 5 minutes limit.
5. **Evaluate:** Achieved **73.33% MacroAccuracy**.
6. **Generate Code:** Prediction code auto-generated, model tested with various samples.

## Limitations

- Only one training architecture is available for Text Classification in Natural Language Processing (NLP) scenarios inside Model Builder.
- AutoML does not explore multiple trainers for NLP scenarios.