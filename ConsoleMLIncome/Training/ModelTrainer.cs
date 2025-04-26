using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleMLIncome.Training
{
    internal class ModelTrainer
    {
        private readonly MLContext _mlContext;

        public ModelTrainer(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        // Train simplified logistic regression model
        public ITransformer TrainAndCompareModels(IEstimator<ITransformer> dataPrepPipeline, IDataView trainData)
        {
            Console.WriteLine("Training a simplified model:");

            // Use simple call without additional parameters
            var estimator = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression();

            Console.WriteLine("Building a pipeline...");

            // Create of a complete pipeline (pre-processing + training)
            var pipeline = dataPrepPipeline.Append(estimator);

            // Train a model with time dimension
            var stopWatch = new Stopwatch();
            stopWatch.Start();

            Console.WriteLine("Start of operation Fit()...");
            var model = pipeline.Fit(trainData);
            stopWatch.Stop();

            Console.WriteLine("Fit() operation completed.");
            Console.WriteLine($"Time of training: {stopWatch.ElapsedMilliseconds / 1000.0:F2} seconds");

            // Do simple assessment
            Console.WriteLine("Performing model evaluation...");
            var predictions = model.Transform(trainData);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");

            return model;
        }

    }
}
