using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleMLBikes
{
    internal class RentalTypePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedRentalType { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
