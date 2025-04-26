using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleMLIncome.Models
{
    internal class AdultPrediction
    {
        // Predicted value (income >50K or <=50K)
        [ColumnName("PredictedLabel")]
        public bool PredictedIncome { get; set; }

        // Probability that income >50K
        public float Probability { get; set; }

        // The value of the estimate before converting it to probability
        public float Score { get; set; }
    }

}
