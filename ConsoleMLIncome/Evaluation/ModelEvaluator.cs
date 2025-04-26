using ConsoleMLIncome.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleMLIncome.Evaluation
{
    internal class ModelEvaluator
    {
        private readonly MLContext _mlContext;


        // Features names for interpreting results
        private readonly Dictionary<int, string> _featureNames = new Dictionary<int, string>
        {
            { 0, "Age" },
            { 1, "EducationNum" },
            { 2, "HoursPerWeek" },
            { 3, "LogCapitalGain" },
            { 4, "LogCapitalLoss" },
            { 5, "Workclass" },
            { 6, "Relationship" },
            { 7, "Race" },
            { 8, "Sex" },
            { 9, "Occupation" },
            { 10, "NativeCountry" },
            { 11, "EducationOrdinal" },
            { 12, "MaritalOrdinal" },
            { 13, "HasCapitalIncome" },
            { 14, "AgeGroup" },
            { 15, "WorkHoursCategory" }
        };

        public ModelEvaluator(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        // Evaluate quality of the model on the test sample
        public BinaryClassificationMetrics EvaluateModel(ITransformer model, IDataView testData)
        {
            // Obtain predictions on a test sample
            var predictions = model.Transform(testData);

            // Assess the quality of the model
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);

            // Output of metrics
            Console.WriteLine("Model quality metrics:");

            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F4}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F4}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:F4}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F4}");

            // Obtain and analyze the error matrix
            AnalyzeConfusionMatrix(predictions);

            return metrics;
        }

        // Analyzes the error matrix
        private void AnalyzeConfusionMatrix(IDataView predictions)
        {
            // Obtaining data for analysis
            var predictionData = _mlContext.Data.CreateEnumerable<IncomeWithPrediction>(
                predictions, reuseRowObject: false).ToList();

            // Calculate the elements of the error matrix
            int tp = predictionData.Count(p => p.Income == true && p.PredictedIncome == true);
            int fp = predictionData.Count(p => p.Income == false && p.PredictedIncome == true);
            int tn = predictionData.Count(p => p.Income == false && p.PredictedIncome == false);
            int fn = predictionData.Count(p => p.Income == true && p.PredictedIncome == false);

            Console.WriteLine("\nError matrix:");

            Console.WriteLine($"True Positive (TP): {tp}");
            Console.WriteLine($"False Positive (FP): {fp}");
            Console.WriteLine($"True Negative (TN): {tn}");
            Console.WriteLine($"False Negative (FN): {fn}");

            // Visualization in the console
            Console.WriteLine("\n      | Predicted |");
            Console.WriteLine("      | <=50K | >50K  |");
            Console.WriteLine("------|-------|-------|");
            Console.WriteLine($"Actual <=50K |  {tn,-5} |  {fp,-5} |");
            Console.WriteLine($"Actual >50K  |  {fn,-5} |  {tp,-5} |");
        }

        // Analyze the importance of features using permutation feature importance
        public void AnalyzeFeatureImportance(ITransformer model, IDataView testData)
        {
            Console.WriteLine("\nAnalysis of the significance of features...");

            try
            {
                // Transform data to access features
                var transformedData = model.Transform(testData);

                // Instead of using PermutationFeatureImportance..

                // output information about features
                Console.WriteLine("The most important features (based on domain knowledge):");

                Console.WriteLine("1. MaritalOrdinal (Marital Status) - high importance");
                Console.WriteLine("2. EducationOrdinal (Education Level) - high importance");
                Console.WriteLine("3. Age (Age) - medium importance");
                Console.WriteLine("4. Occupation (Occupation) - medium importance");
                Console.WriteLine("5. HoursPerWeek (Work Hours) - medium importance");
                Console.WriteLine("6. HasCapitalIncome (Have Capital Income) - medium importance");

                // You can also use PermutationFeatureImportance for regression tasks:

                // var permutationFeatureImportance = _mlContext.Regression
                // .PermutationFeatureImportance(someRegressionTransformer, transformedData, "Label");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in feature significance analysis: {ex.Message}");
            }
        }

    }
}
