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
    /// Parses a csv file in its entirety as data. Assumes no label is present and all columns will be data points. Useful for test.csv files (which usually do not contain labels).
    /// </summary>
    public class TestParser : BaseParser
    {
        public override int ReadLabel(CsvReader reader)
        {
            return 0;
        }

        public override List<double> ReadData(CsvReader reader)
        {
            return ReadData(reader, 0);
        }
    }
}
