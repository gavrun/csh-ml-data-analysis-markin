using Microsoft.ML;

namespace ConsoleMLBikes
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, Data Analysis!");

            // Context 
            var mlContext = new MLContext(seed: 0);

            // Dataset 
            //string dataPath = "data/bike_sharing.csv"; // Copy to Output Directory
            string dataPath = "data/bike_sharing_200.csv"; 
            //string dataPath = Path.Combine(AppContext.BaseDirectory, @"..\..\..\data\bike_sharing_200.csv");


            // Load data to ML data structure
            IDataView data = mlContext.Data.LoadFromTextFile<BikeRentalData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            // Split data into training and testing sets
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.1);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            Console.WriteLine($"Train data loaded and split.");

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
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "RentalType",
                featureColumnName: "Features");

            // Data data processing and data training (single pipeline)
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train model
            var trainedModel = trainingPipeline.Fit(trainData);

            Console.WriteLine("Model trained.");


            // Predictions by set
            var predictions = trainedModel.Transform(testData);


            // Evaluate model performance
            var metrics = mlContext.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "RentalType",
                scoreColumnName: "Score");

            Console.WriteLine($"The model evaluated on test data:");
            Console.WriteLine($"  Accuracy:  {metrics.Accuracy:P2}");
            Console.WriteLine($"  AUC:       {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  F1 Score:  {metrics.F1Score:P2}");

            // Prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(trainedModel);

            // test single input
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

            Console.WriteLine("Example prediction:");
            Console.WriteLine($"  Long term lease probability: {prediction.Probability:P2}");
            Console.WriteLine($"  Predicted rental type: {(prediction.PredictedRentalType ? "Long term" : "Short term")}");

        }
    }
}
