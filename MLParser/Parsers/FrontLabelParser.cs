using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using MLParser.Interface;
using MLParser.Types;

namespace MLParser.Parsers
{
    /// <summary>
    /// Parses a csv file, assuming column 0 contains the label and the remaining columns contain the data.
    /// </summary>
    public class FrontLabelParser : BaseParser
    {
        public override int ReadLabel(CsvReader reader)
        {
            return Int32.Parse(reader[0]);
        }

        public override List<double> ReadData(CsvReader reader)
        {
            // Start at index 1, as the index 0 contains the label.
            return ReadData(reader, 1);
        }
    }
}
