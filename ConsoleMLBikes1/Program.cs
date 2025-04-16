using Microsoft.ML;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Xml.Linq;

namespace ConsoleMLBikes
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, Data Analysis!\n");

            // Context 
            var mlContext = new MLContext(seed: 0);

            // Dataset 
            //string dataPath = "Data/bike_sharing.csv"; // Copy to Output Directory
            string dataPath = "Data/bike_sharing_1000.csv";


            // Load data to ML data structure
            IDataView data = mlContext.Data.LoadFromTextFile<BikeRentalData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            // Split data into training and testing sets
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.1);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            Console.WriteLine($"\nTrain data loaded and split.");

            Console.WriteLine($"DEBUG: train count: {mlContext.Data.CreateEnumerable<BikeRentalData>(trainData, reuseRowObject: false).Count()}");
            Console.WriteLine($"DEBUG: test count:  {mlContext.Data.CreateEnumerable<BikeRentalData>(testData, reuseRowObject: false).Count()}");


            // Data preprocessing pipeline

            // Encode categorical fields
            // Normalize numerical fields
            // Combine all features into vector
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new[]
                {
                    new InputOutputColumnPair("SeasonEncoded", "Season"),
                    new InputOutputColumnPair("WeatherEncoded", "WeatherCondition")
                })
                .Append(mlContext.Transforms.NormalizeMinMax("Temperature"))
                .Append(mlContext.Transforms.NormalizeMinMax("Humidity"))
                .Append(mlContext.Transforms.NormalizeMinMax("Windspeed"))
                .Append(mlContext.Transforms.Concatenate("Features", "SeasonEncoded", "WeatherEncoded", "Month", "Hour", "Holiday", "Weekday", "WorkingDay", "Temperature", "Humidity", "Windspeed")
                );

            // Training algorithm
            //var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            //    labelColumnName: "RentalType",
            //    featureColumnName: "Features");

            // Data data processing and data training (single pipeline)
            //var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train model
            //var trainedModel = trainingPipeline.Fit(trainData);

            //Console.WriteLine("Model trained.");


            // Saved trained model
            ITransformer savedModel;
            string modelDirectory = "Models";
            //string modelPath = "Models/model.zip";
            string modelPath = Path.Combine(modelDirectory, "model.zip");

            PredictionEngine<BikeRentalData, RentalTypePrediction> predictionEngine;


            // Training model set
            var models = new List<(string name, IEstimator<ITransformer> trainer)>
            {
                ("LogisticRegression", mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "RentalType", featureColumnName: "Features")),
                ("FastTree", mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "RentalType", featureColumnName: "Features")),
                ("LightGbm", mlContext.BinaryClassification.Trainers.LightGbm(
                    labelColumnName: "RentalType", featureColumnName: "Features"))
            };

            // Predictions by set
            //var predictions = trainedModel.Transform(testData);


            // Evaluate model performance
            //var metrics = mlContext.BinaryClassification.Evaluate(
            //    data: predictions,
            //    labelColumnName: "RentalType",
            //    scoreColumnName: "Score");

            //Console.WriteLine($"The model evaluated on test data:");
            //Console.WriteLine($"  Accuracy:  {metrics.Accuracy:P2}");
            //Console.WriteLine($"  AUC:       {metrics.AreaUnderRocCurve:P2}");
            //Console.WriteLine($"  F1 Score:  {metrics.F1Score:P2}");


            // Evaluate performance in model set

            if (File.Exists(modelPath))
            {
                Console.WriteLine("\nLoading model from disk");
                savedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

                predictionEngine = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(savedModel);
            }
            else
            {
                Console.WriteLine("\nNo saved models.");

                ITransformer bestModel = null;
                string bestModelName = "";
                double bestF1 = 0;

                foreach (var (_name, _trainer) in models)
                {
                    Console.WriteLine($"\nTraining model: {_name}");

                    var _pipeline = dataProcessPipeline.Append(_trainer);

                    var _model = _pipeline.Fit(trainData);

                    var _predictions = _model.Transform(testData);

                    var _metrics = mlContext.BinaryClassification.Evaluate(_predictions, labelColumnName: "RentalType");

                    Console.WriteLine($"  Accuracy:  {_metrics.Accuracy:P2}");
                    Console.WriteLine($"  AUC:       {_metrics.AreaUnderRocCurve:P2}");
                    Console.WriteLine($"  F1 Score:  {_metrics.F1Score:P2}");

                    // Save the best model based on F1 Score
                    if (_metrics.F1Score > bestF1)
                    {
                        bestF1 = _metrics.F1Score;
                        bestModel = _model;
                        bestModelName = _name;
                    }
                }

                // Save best model
                if (!Directory.Exists(modelDirectory)) 
                {
                    Directory.CreateDirectory(modelDirectory);
                }    
                mlContext.Model.Save(bestModel, trainData.Schema, modelPath);
                Console.WriteLine("\nBest model saved.");

                predictionEngine = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(bestModel);
            }

            // Prediction engine
            //var predictionEngine = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(trainedModel);
            

            // Test single input
            var sample = new BikeRentalData
            {
                Season = 2,
                Month = 4,
                Hour = 10,
                Holiday = 0,
                Weekday = 3,
                WorkingDay = 1,
                WeatherCondition = 1,
                Temperature = 18,
                Humidity = 55,
                Windspeed = 12,
            };

            // Make prediction by sample
            var prediction = predictionEngine.Predict(sample);

            Console.WriteLine("\nExample prediction:");
            Console.WriteLine($"  Long term lease probability: {prediction.Probability:P2}");
            Console.WriteLine($"  Predicted rental type: {(prediction.PredictedRentalType ? "Long term" : "Short term")}");

        }
    }
}
