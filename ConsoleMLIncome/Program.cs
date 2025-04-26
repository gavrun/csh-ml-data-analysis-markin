using ConsoleMLIncome.DataProcessing;
using ConsoleMLIncome.Training;
using Microsoft.ML;

namespace ConsoleMLIncome
{
    internal class Program
    {
        // Path to the dataset
        private static string _dataPath = "Data/adult.csv";

        static void Main(string[] args)
        {
            Console.WriteLine("Hello, ML!");

            Console.WriteLine("\nForecasting Person's Income Based on Demographic Data");

            // Create a ML.NET context with a fixed seed for reproducible results
            var mlContext = new MLContext(seed: 42);

            // Data preprocessing pipeline
            try
            {
                // Stage 1. Load and analyze
                Console.WriteLine("\nStep 1: Load and analyze data...");
                var dataProcessor = new DataProcessor(mlContext);
                var data = dataProcessor.LoadData(_dataPath);

                var sampleData = mlContext.Data.TakeRows(data, 1000);
                dataProcessor.ExploreData(sampleData);

                // Stage 2. Prepare and create pipeline
                Console.WriteLine("\nStep 2: Split data and create processing pipeline...");
                var trainTestData = dataProcessor.SplitData(sampleData);
                var dataPrepPipeline = dataProcessor.CreateDataProcessingPipeline();

                // Stage 3. Train
                Console.WriteLine("\nStep 3: Train model...");
                var modelTrainer = new ModelTrainer(mlContext);
                var model = modelTrainer.TrainAndCompareModels(dataPrepPipeline, trainTestData.TrainSet);

                Console.WriteLine("\nTraining completed successfully!");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nERROR: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
                //throw;
            }

            Console.WriteLine("\nPress any key to finish...");
            Console.ReadKey();
        }
    }
}
