using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLParser.Types
{
    /// <summary>
    /// Data-type for holding machine learning data from a csv file, consisting of an array of doubles (input) and a label (output).
    /// </summary>
    public class MLData
    {
        /// <summary>
        /// Input
        /// </summary>
        public List<double> Data { get; set; }
        /// <summary>
        /// Output
        /// </summary>
        public int Label { get; set; }

        public MLData()
        {
            Data = new List<double>();
        }
    }
}
