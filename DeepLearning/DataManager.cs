using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning
{
    public static class DataManager
    {
        public static double[][] Load(string pathName, out double[][] outputs)
        {
            List<double[]> list = new List<double[]>();
            List<double[]> output = new List<double[]>();

            // Read data file.
            using (FileStream fs = File.Open(pathName, FileMode.Open, FileAccess.Read))
            {
                using (BufferedStream bs = new BufferedStream(fs))
                {
                    using (StreamReader sr = new StreamReader(bs))
                    {
                        List<double> row = new List<double>();

                        bool readOutput = false;

                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            // Collect each 0 and 1 from the data.
                            foreach (char ch in line)
                            {
                                if (!readOutput)
                                {
                                    // Reading input.
                                    if (ch != ' ' && ch != '\n')
                                    {
                                        // Add this digit to our input.
                                        row.Add(Double.Parse(ch.ToString()));
                                    }
                                    else if (ch == ' ')
                                    {
                                        // End of input reached. Store the input row.
                                        list.Add(row.ToArray());

                                        // Start a new input row.
                                        row = new List<double>();

                                        // Set flag to read output label.
                                        readOutput = true;
                                    }
                                }
                                else
                                {
                                    // Read output label.
                                    output.Add(FormatOutputVector(Double.Parse(ch.ToString())));

                                    // Set flag to read inputs for next row.
                                    readOutput = false;
                                }
                            }
                        }
                    }
                }
            }

            // Set outputs.
            outputs = output.ToArray();

            // Return inputs;
            return list.ToArray();
        }

        #region Utility Methods

        /// <summary>
        /// Converts a numeric output label (0, 1, 2, 3, etc) to its cooresponding array of doubles, where all values are 0 except for the index matching the label (ie., if the label is 2, the output is [0, 0, 1, 0, 0, ...]).
        /// </summary>
        /// <param name="label">double</param>
        /// <returns>double[]</returns>
        public static double[] FormatOutputVector(double label)
        {
            double[] output = new double[10];

            for (int i = 0; i < output.Length; i++)
            {
                if (i == label)
                {
                    output[i] = 1;
                }
                else
                {
                    output[i] = 0;
                }
            }

            return output;
        }

        /// <summary>
        /// Finds the largest output value in an array and returns its index. This allows for sequential classification from the outputs of a neural network (ie., if output at index 2 is the largest, the classification is class "3" (zero-based)).
        /// </summary>
        /// <param name="output">double[]</param>
        /// <returns>double</returns>
        public static double FormatOutputResult(double[] output)
        {
            return output.ToList().IndexOf(output.Max());
        }

        #endregion
    }
}
