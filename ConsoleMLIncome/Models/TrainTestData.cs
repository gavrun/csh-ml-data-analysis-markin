using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleMLIncome.Models
{
    internal class TrainTestData
    {
        // Class for storing separated training and test samples

        public IDataView TrainSet { get; set; }

        public IDataView TestSet { get; set; }

        public TrainTestData(IDataView trainSet, IDataView testSet)
        {
            TrainSet = trainSet;
            TestSet = testSet;
        }

    }
}
