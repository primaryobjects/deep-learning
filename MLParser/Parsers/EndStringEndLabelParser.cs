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
    /// Parses a csv file, assuming the last 2 columns consist of a string (filename) followed by the label, and the remaining columns contain the data.
    /// </summary>
    public class EndStringEndLabelParser : BaseParser
    {
        public override int ReadLabel(CsvReader reader)
        {
            return Int32.Parse(reader[reader.Parser.FieldCount - 1]);
        }

        public override List<double> ReadData(CsvReader reader)
        {
            // Start at index 0, and read up to the last 2 columns, which are the string (filename) and label.
            return ReadData(reader, 0, reader.Parser.FieldCount - 2);
        }
    }
}
