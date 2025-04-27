using TorchSharp.Modules;

namespace ConsoleMLSpam
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, ML!");

            // Samples input

            var sampleData1 = new MLModel.ModelInput()
            {
                Content = @"Hi, Bill! Congratulations! You've won a free cruise to the Bahamas! Click on this link now to claim your tickets. Greetings, TourismTeam"
            };
            var sampleData2 = new MLModel.ModelInput()
            {
                Content = @"Hi team, Just a reminder that our weekly team meeting is scheduled for Thursday at 2 PM in Conference Room A. Please bring your project updates. Thanks, Greg Bishop, Regional Manager"
            };
            var sampleData3 = new MLModel.ModelInput()
            {
                Content = @"URGENT! Special promotion: All employees get 20% off on office supplies this week. Visit the internal company store for more details. Follow the link below."
            };

            //List<MLModel.ModelInput> samples = [ sampleData1, sampleData2, sampleData3 ];

            List<MLModel.ModelInput> samples = new List<MLModel.ModelInput>()
            {
                sampleData1, sampleData2, sampleData3
            };

            // Model: TextClassificationMulti [TorchSharp] NLP support

            // Predict using the trained model
            foreach (var sample in samples)
            {
                var predictionResult = MLModel.Predict(sample);

                // Score[0] = not spam probability
                // Score[1] = spam probability
                bool isSpam = predictionResult.Score[1] > 0.5;

                Console.WriteLine($"\nSample Text: {sample.Content}");

                // Output prediction
                Console.WriteLine($"\nPredicted label (IsSpam): {predictionResult.PredictedLabel}"); // ISSUE
                Console.WriteLine($"\nDEBUG Calculated label: {isSpam}"); 
                
                Console.WriteLine($"Probability to be valid email: {predictionResult.Score[0]:P2}");
                Console.WriteLine($"Probability to be a spam: {predictionResult.Score[1]:P2}\n");
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}
