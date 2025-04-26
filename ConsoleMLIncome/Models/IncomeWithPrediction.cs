using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleMLIncome.Models
{
    internal class IncomeWithPrediction
    {
        // Actual value of income
        public bool Income { get; set; }

        // Predicted Income Value
        public bool PredictedIncome { get; set; }

        // Probability of prediction
        public float Probability { get; set; }
    }

}
