using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleMLIncome.Models
{
    internal class CustomMapping
    {

    }

    // Helper class for capital transformation (emissions handling)
    public class CapitalTransformedData
    {
        public float LogCapitalGain { get; set; }
        public float LogCapitalLoss { get; set; }
    }

    // Helper class for ordinal coding of education
    public class EducationMappingData
    {
        public float EducationOrdinal { get; set; }
    }

    // Helper class for ordinal coding of marital status
    public class MaritalMappingData
    {
        public float MaritalOrdinal { get; set; }
    }

    // Helper class for creating derived features
    public class DerivedFeaturesData
    {
        public float HasCapitalIncome { get; set; }
        public float AgeGroup { get; set; }
        public float WorkHoursCategory { get; set; }
    }

    // Class for representing the transformed dataset
    public class TransformedAdultData
    {
        public float Age { get; set; }
        public float EducationNum { get; set; }
        public float HoursPerWeek { get; set; }
        public float LogCapitalGain { get; set; }
        public float LogCapitalLoss { get; set; }
        public float[]? WorkclassEncoded { get; set; }  // nullable
        public float EducationOrdinal { get; set; }
        public float MaritalOrdinal { get; set; }
        public float HasCapitalIncome { get; set; }
        public float AgeGroup { get; set; }
        public float WorkHoursCategory { get; set; }
        public float[]? Features { get; set; }  // nullable
        public bool Label { get; set; }
    }

    // Class for converting string Income to boolean value
    public class AdultDataWithBoolLabel
    {
        [ColumnName("Label")]
        public bool Label { get; set; }
    }
}
