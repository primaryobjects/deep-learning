using CsvHelper;
using MLParser.Interface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLParser.Parsers
{
    public abstract class BaseParser : IRowParser
    {
        public abstract int ReadLabel(CsvReader reader);
        public abstract List<double> ReadData(CsvReader reader);

        /// <summary>
        /// Helper method for reading a row of data from a csv file. Reading starts at the startColumn and ends at the endColumn.
        /// </summary>
        /// <param name="reader">CsvReader</param>
        /// <param name="startColumn">int - start index to begin reading fields from.</param>
        /// <param name="endColumn">int - end index to stop reading fields at. Set to null to read until the end of the row.</param>
        /// <returns>List of double</returns>
        protected List<double> ReadData(CsvReader reader, int startColumn, int? endColumn = null)
        {
            List<double> data = new List<double>();

            if (endColumn == null)
            {
                // Read until the end of the row.
                endColumn = reader.Parser.FieldCount;
            }

            // Start at index to begin reading data from.
            for (int i = startColumn; i < endColumn; i++)
            {
                // Read the value.
                double value = Double.Parse(reader[i]);

                // Store the normalized value in our data list.
                data.Add(Normalize(value));
            }

            return data;
        }

        protected double Normalize(double value)
        {
            // Normalize the value (0 - 1): X = (X - min) / (max - min) => X = X / 255. Alternate method (-0.5 - 0.5): X = (X - avg) / max - min => X = (X - 127) / 255. http://en.wikipedia.org/wiki/Feature_scaling
            return value / 255d;
        }
    }
}
