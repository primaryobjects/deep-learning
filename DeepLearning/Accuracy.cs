using Accord.MachineLearning.VectorMachines;
using Accord.Neuro.Networks;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning
{
    public static class Accuracy
    {
        /// <summary>
        /// Calculates the accuracy for a trainined SVM against the data.
        /// </summary>
        /// <param name="machine">MulticlassSupportVectorMachine</param>
        /// <param name="data">List of DigitData</param>
        /// <returns>double</returns>
        public static double CalculateAccuracy(MulticlassSupportVectorMachine machine, double[][] inputs, int[] outputs)
        {
            double correct = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                int output = machine.Compute(inputs[i]);
                if (output == outputs[i])
                {
                    correct++;
                }
            }

            return (correct / (double)inputs.Length);
        }

        /// <summary>
        /// Calculates the accuracy for a trainined DNN against the data.
        /// </summary>
        /// <param name="network">DeepBeliefNetwork</param>
        /// <param name="data">List of DigitData</param>
        /// <returns>double</returns>
        public static double CalculateAccuracy(DeepBeliefNetwork network, double[][] inputs, double[][] outputs)
        {
            double correct = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] outputValues = network.Compute(inputs[i]);
                if (DataManager.FormatOutputResult(outputValues) == DataManager.FormatOutputResult(outputs[i]))
                {
                    correct++;
                }
            }

            return (correct / (double)inputs.Length);
        }

        /// <summary>
        /// Calculates the output for the svm and saves each result to a text file, one per line.
        /// </summary>
        /// <param name="machine">MulticlassSupportVectorMachine</param>
        /// <param name="inputs">double[][]</param>
        /// <param name="path">string</param>
        /// <returns>int - number of rows processed</returns>
        public static int SaveOutput(MulticlassSupportVectorMachine machine, double[][] inputs, string path)
        {
            File.AppendAllText(path, "ImageId,Label\r\n");

            for (int i = 0; i < inputs.Length; i++)
            {
                int output = machine.Compute(inputs[i]);
                File.AppendAllText(path, (i + 1) + "," + output.ToString() + "\r\n");
            }

            return inputs.Length;
        }

        /// <summary>
        /// Calculates the output for the neural network and saves each result to a text file, one per line.
        /// </summary>
        /// <param name="machine">DeepBeliefNetwork</param>
        /// <param name="inputs">double[][]</param>
        /// <param name="path">string</param>
        /// <returns>int - number of rows processed</returns>
        public static int SaveOutput(DeepBeliefNetwork network, double[][] inputs, string path)
        {
            File.AppendAllText(path, "ImageId,Label\r\n");

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] outputValues = network.Compute(inputs[i]);
                double output = DataManager.FormatOutputResult(outputValues);
                
                File.AppendAllText(path, (i + 1) + "," + output.ToString() + "\r\n");
            }

            return inputs.Length;
        }
    }
}
