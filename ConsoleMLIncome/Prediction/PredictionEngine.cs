using ConsoleMLIncome.Models;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleMLIncome.Prediction
{
    internal class PredictionEngine
    {
        private readonly MLContext _mlContext;
        
        private ITransformer? _loadedModel;  // nullable
        
        private PredictionEngine<AdultData, AdultPrediction>? _predictionEngine;  // nullable

        public PredictionEngine(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        // Loads a model from a file
        public void LoadModel(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Model file not found: {modelPath}");
            }

            // Loading model
            DataViewSchema modelSchema;
            _loadedModel = _mlContext.Model.Load(modelPath, out modelSchema);

            // Creating a forecasting engine
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<AdultData, AdultPrediction>(_loadedModel);

            Console.WriteLine($"Model loaded from file: {modelPath}");
        }

        // Forecasts using predefined examples
        public void DemonstratePredictions()
        {
            Console.WriteLine("Demonstration of predictions using typical examples:");

            // Example 1: Highly paid specialist
            var example1 = new AdultData
            {
                Age = 45,
                Workclass = "Private",
                Education = "Bachelors",
                EducationNum = 13,
                MaritalStatus = "Married-civ-spouse",
                Occupation = "Exec-managerial",
                Relationship = "Husband",
                Race = "White",
                Sex = "Male",
                CapitalGain = 15000,
                CapitalLoss = 0,
                HoursPerWeek = 60,
                NativeCountry = "United-States"
            };

            var prediction1 = _predictionEngine.Predict(example1);
            Console.WriteLine("\nExample 1: Highly paid specialist");
            Console.WriteLine("Characteristics:");
            Console.WriteLine("- 45 years old, male, married");
            Console.WriteLine("- Higher education (bachelor's)");
            Console.WriteLine("- Management position in a private company");
            Console.WriteLine("- Capital increase: $15,000");
            Console.WriteLine("- 60 hours of work per week");
            Console.WriteLine($"Prediction: {(prediction1.PredictedIncome ? ">50K" : "<=50K")}");
            Console.WriteLine($"Probability of income >50K: {prediction1.Probability:P2}");

            // Example 2: Low-wage worker
            var example2 = new AdultData
            {
                Age = 25,
                Workclass = "Private",
                Education = "HS-grad",
                EducationNum = 9,
                MaritalStatus = "Never-married",
                Occupation = "Service",
                Relationship = "Not-in-family",
                Race = "Black",
                Sex = "Female",
                CapitalGain = 0,
                CapitalLoss = 0,
                HoursPerWeek = 35,
                NativeCountry = "United-States"
            };

            var prediction2 = _predictionEngine.Predict(example2);
            Console.WriteLine("\nExample 2: Low-wage worker");
            Console.WriteLine("Characteristics:");
            Console.WriteLine("- 25 years old, female, single");
            Console.WriteLine("- Secondary education");
            Console.WriteLine("- Service sector in a private company");
            Console.WriteLine("- No capital gain");
            Console.WriteLine("- 35 hours of work per week");
            Console.WriteLine($"Prediction: {(prediction2.PredictedIncome ? ">50K" : "<=50K")}");
            Console.WriteLine($"Probability of income >50K: {prediction2.Probability:P2}");
        }


        // Launches interactive prediction mode
        public void RunInteractivePredictions()
        {
            Console.WriteLine("\nInteractive Income Forecasting.");
            Console.WriteLine("Enter data to forecast or 'exit' to exit\n");

            while (true)
            {
                Console.WriteLine("\nEnter information about the person:");

                Console.Write("AGe (example, 35): ");
                string ageInput = Console.ReadLine();
                if (string.Equals(ageInput, "exit", StringComparison.OrdinalIgnoreCase))
                    break;

                Console.Write("Class of work (example, Private, Self-emp, Federal-gov): ");
                string workclass = Console.ReadLine();

                Console.Write("Education (example, Bachelors, HS-grad, Masters): ");
                string education = Console.ReadLine();

                Console.Write("Number of years of education (example, 13): ");
                string eduNumInput = Console.ReadLine();

                Console.Write("Marital status (example, Married-civ-spouse, Never-married): ");
                string maritalStatus = Console.ReadLine();

                Console.Write("Occupation (example, Exec-managerial, Prof-specialty): ");
                string occupation = Console.ReadLine();

                Console.Write("Gender (Male/Female): ");
                string sex = Console.ReadLine();

                Console.Write("Hours of work per week (example, 40): ");
                string hoursInput = Console.ReadLine();

                Console.Write("Capital gain ($): ");
                string capitalGainInput = Console.ReadLine();

                Console.Write("Capital losses ($): ");
                string capitalLossInput = Console.ReadLine();

                // Creating an object with data
                var inputData = new AdultData
                {
                    Age = float.TryParse(ageInput, out var age) ? age : 0,
                    Workclass = workclass,
                    Education = education,
                    EducationNum = float.TryParse(eduNumInput, out var eduNum) ? eduNum : 0,
                    MaritalStatus = maritalStatus,
                    Occupation = occupation,
                    Sex = sex,
                    HoursPerWeek = float.TryParse(hoursInput, out var hours) ? hours : 0,
                    CapitalGain = float.TryParse(capitalGainInput, out var gain) ? gain : 0,
                    CapitalLoss = float.TryParse(capitalLossInput, out var loss) ? loss : 0,
                    // Остальные поля заполняем значениями по умолчанию
                    Relationship = "Unknown",
                    Race = "Unknown",
                    NativeCountry = "United-States"
                };

                // Receiving a prediction
                var prediction = _predictionEngine.Predict(inputData);

                // Output of the result
                Console.WriteLine("\nPrediction Result");
                Console.WriteLine($"Predicted Income Level: {(prediction.PredictedIncome ? ">50K" : "<=50K")}");
                Console.WriteLine($"Probability of Income >50K: {prediction.Probability:P2}");
                Console.WriteLine($"Model Confidence: {(prediction.Probability > 0.5 ? prediction.Probability : 1 - prediction.Probability):P2}");
            }
        }

    }
}
