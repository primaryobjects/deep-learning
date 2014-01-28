using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Neuro.Learning;
using System.IO;

namespace DeepLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            // We'll use a simple XOR function as input.
            double[][] inputs = new double[][] { 
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            // XOR output, cooresponding to the input.
            double[][] outputs = new double[][] {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            // Setup the deep belief network and initialize with random weights.
            DeepBeliefNetwork network = new DeepBeliefNetwork(inputs.First().Length, 1, 1);
            new GaussianWeights(network, 0.1).Randomize();
            network.UpdateVisibleWeights();

            // Setup the learning algorithm.
            DeepBeliefNetworkLearning teacher = new DeepBeliefNetworkLearning(network)
            {
                Algorithm = (h, v, i) => new ContrastiveDivergenceLearning(h, v)
                {
                    LearningRate = 0.1,
                    Momentum = 0.5,
                    Decay = 0.001,
                }
            };

            // Setup batches of input for learning.
            int batchCount = Math.Max(1, inputs.Length / 100);
            // Create mini-batches to speed learning.
            int[] groups = Accord.Statistics.Tools.RandomGroups(inputs.Length, batchCount);
            double[][][] batches = inputs.Subgroups(groups);
            // Learning data for the specified layer.
            double[][][] layerData;

            // Unsupervised learning on each hidden layer, except for the output.
            for (int layerIndex = 0; layerIndex < network.Machines.Count - 1; layerIndex++)
            {
                teacher.LayerIndex = layerIndex;
                layerData = teacher.GetLayerInput(batches);
                for (int i = 0; i < 500; i++)
                {
                    double error = teacher.RunEpoch(layerData) / inputs.Length;
                    if (i % 10 == 0)
                    {
                        Console.WriteLine(i + ", Error = " + error);
                    }
                }
            }

            // Supervised learning on entire network, to provide output classification.
            var teacher2 = new BackPropagationLearning(network)
            {
                LearningRate = 0.1,
                Momentum = 0.5
            };

            // Run supervised learning.
            for (int i = 0; i < 5000; i++)
            {
                double error = teacher2.RunEpoch(inputs, outputs) / inputs.Length;
                if (i % 10 == 0)
                {
                    Console.WriteLine(i + ", Error = " + error);
                }
            }

            // Test the resulting accuracy.
            int correct = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] outputValues = network.Compute(inputs[i]);
                if (FormatOutputResult(outputValues) == FormatOutputResult(outputs[i]))
                {
                    correct++;
                }
            }

            Console.WriteLine("Correct " + correct + "/" + inputs.Length + ", " + Math.Round(((double)correct / (double)inputs.Length * 100), 2) + "%");
            Console.Write("Press any key to quit ..");
            Console.ReadKey();
        }

        #region Utility Methods

        /// <summary>
        /// Converts a numeric output label (0, 1, 2, 3, etc) to its cooresponding array of doubles, where all values are 0 except for the index matching the label (ie., if the label is 2, the output is [0, 0, 1, 0, 0, ...]).
        /// </summary>
        /// <param name="label">double</param>
        /// <returns>double[]</returns>
        private static double[] FormatOutputVector(double label)
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
        private static double FormatOutputResult(double[] output)
        {
            return output.ToList().IndexOf(output.Max());
        }

        #endregion
    }
}
