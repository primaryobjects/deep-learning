using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using MLParser.Interface;
using MLParser.Types;

namespace MLParser
{
    public class Parser
    {
        private IRowParser _rowParser = null;

        public Parser(IRowParser rowParser)
        {
            _rowParser = rowParser;
        }

        /// <summary>
        /// Parses a csv file containing inputs and an output label, returning a list of MLData.
        /// </summary>
        /// <param name="path">string</param>
        /// <param name="maxRows">int - max number of rows to read</param>
        /// <returns>List of MLData</returns>
        public List<MLData> Parse(string path, int maxRows = 0)
        {
            List<MLData> dataList = new List<MLData>();

            using (FileStream f = new FileStream(path, FileMode.Open))
            {
                using (StreamReader streamReader = new StreamReader(f, Encoding.GetEncoding(1252)))
                {
                    using (CsvReader csvReader = new CsvReader(streamReader))
                    {
                        csvReader.Configuration.HasHeaderRecord = false;

                        while (csvReader.Read())
                        {
                            MLData row = new MLData()
                            {
                                Label = _rowParser.ReadLabel(csvReader),
                                Data = _rowParser.ReadData(csvReader)
                            };

                            dataList.Add(row);

                            if (maxRows > 0 && dataList.Count >= maxRows)
                                break;
                        }
                    }
                }
            }

            return dataList;
        }
    }
}
