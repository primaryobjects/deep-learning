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
using MLParser;
using MLParser.Parsers;
using MLParser.Types;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using MLParser.Interface;
using System.Diagnostics;
using System.Configuration;

namespace DeepLearning
{
    class Program
    {
        #region App.Config Values

        private static int _pixelCount = Int32.Parse(ConfigurationManager.AppSettings["Width"]) * Int32.Parse(ConfigurationManager.AppSettings["Height"]);
        private static int _classCount = Int32.Parse(ConfigurationManager.AppSettings["ClassCount"]);
        private static int _trainCount = Int32.Parse(ConfigurationManager.AppSettings["TrainCount"]);
        private static int _hiddenEpochCount = Int32.Parse(ConfigurationManager.AppSettings["HiddenEpochCount"]);
        private static int _fineTuneEpochCount = Int32.Parse(ConfigurationManager.AppSettings["FineTuneEpochCount"]);
        private static int _hiddenNeuronCount = Int32.Parse(ConfigurationManager.AppSettings["HiddenNeuronCount"]);
        private static double _sigma = Double.Parse(ConfigurationManager.AppSettings["Sigma"]);
        private static string _trainPath = ConfigurationManager.AppSettings["TrainPath"];
        private static string _cvPath = ConfigurationManager.AppSettings["CvPath"];
        private static string _testPath = ConfigurationManager.AppSettings["TestPath"];

        #endregion

        static void Main(string[] args)
        {
            Console.WriteLine("-= Training =-");
            var network = RunDNN(_trainPath, _trainCount, _hiddenEpochCount, _fineTuneEpochCount);
            Console.WriteLine("-= Cross Validation =-");
            RunDNN(_cvPath, _trainCount, _hiddenEpochCount, _fineTuneEpochCount, network);

            /*for (int count = 200; count < 4000; count += 200)
            {
                Console.WriteLine("-= Training =-");
                var machine = RunSvm(@"../../../data/catsdogs-train.csv", count);

                Console.WriteLine("-= Cross Validation =-");
                RunSvm(@"../../../data/catsdogs-cv.csv", count, machine);
            }*/

            Console.Write("Press any key to quit ..");
            Console.ReadKey();
        }

        /// <summary>
        /// Core machine learning method for parsing csv data, training the network, and calculating the accuracy.
        /// </summary>
        /// <param name="path">string - path to csv file (training, csv, test).</param>
        /// <param name="count">int - max number of rows to process. This is useful for preparing learning curves, by using gradually increasing values. Use 0 to read all rows.</param>
        /// <param name="hiddenEpochs">int - max number of epochs per hidden layer (unsupervised).</param>
        /// <param name="fineTuneEpochs">int - max number of epochs over entire network during fine-tuning (supervised). Set to 0 to be the same as hiddenEpochs.</param>
        /// <param name="network">DeepBeliefNetwork - Leave null for initial training.</param>
        /// <returns>DeepBeliefNetwork</returns>
        private static DeepBeliefNetwork RunDNN(string path, int count, int hiddenEpochs, int fineTuneEpochs = 0, DeepBeliefNetwork network = null)
        {
            double[][] inputs;
            double[][] outputs;
            int[] intOutputs;

            if (fineTuneEpochs == 0)
            {
                fineTuneEpochs = hiddenEpochs;
            }

            // Parse the csv file to get inputs and outputs.
            ReadData(path, count, out inputs, out intOutputs, new EndStringEndLabelParser());
            
            // Format output as double[][].
            outputs = intOutputs.Select(o => DataManager.FormatOutputVector((double)o, _classCount)).ToArray();

            if (network == null)
            {
                // Training.
                network = new DeepBeliefNetwork(inputs.First().Length, _hiddenNeuronCount, _hiddenNeuronCount, _hiddenNeuronCount, _hiddenNeuronCount, outputs.First().Length);
                new NguyenWidrow(network).Randomize();
                network.UpdateVisibleWeights();
                network.Save(@"../../../data/network1.dat");

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

                DateTime startTime = DateTime.Now;
                DateTime epochStart = DateTime.Now;

                // Unsupervised learning on each hidden layer, except for the output layer.
                for (int layerIndex = 0; layerIndex < network.Machines.Count - 1; layerIndex++)
                {
                    teacher.LayerIndex = layerIndex;
                    layerData = teacher.GetLayerInput(batches);
                    for (int i = 0; i < hiddenEpochs; i++)
                    {
                        double error = teacher.RunEpoch(layerData) / inputs.Length;
                        if (i % 2 == 0)
                        {
                            TimeSpan timeSpan = DateTime.Now - epochStart;
							int epochsRemainingByCount = (int)Math.Round((double)(hiddenEpochs - i) / (double)2);
							double minutesRemaining = epochsRemainingByCount * timeSpan.TotalMinutes;
							DateTime finishTime = DateTime.Now + TimeSpan.FromMinutes(minutesRemaining);
							
                            Console.WriteLine(i + ", Error = " + error + ", " + Math.Round(timeSpan.TotalMinutes) + "m (" + Math.Round(timeSpan.TotalSeconds) + "s), eta " + finishTime.ToShortTimeString());
                            epochStart = DateTime.Now;
                        }
                    }
                }

                network.Save(@"../../../data/network2.dat");

                // Supervised learning on entire network, to provide output classification.
                var teacher2 = new BackPropagationLearning(network)
                {
                    LearningRate = 0.1,
                    Momentum = 0.5
                };

                epochStart = DateTime.Now;

                // Run supervised learning.
                for (int i = 0; i < fineTuneEpochs; i++)
                {
                    double error = teacher2.RunEpoch(inputs, outputs) / inputs.Length;
                    if (i % 2 == 0)
                    {
                        TimeSpan timeSpan = DateTime.Now - epochStart;
                        int epochsRemainingByCount = (int)Math.Round((double)(fineTuneEpochs - i) / (double)2);
                        double minutesRemaining = epochsRemainingByCount * timeSpan.TotalMinutes;
                        DateTime finishTime = DateTime.Now + TimeSpan.FromMinutes(minutesRemaining);

                        Console.WriteLine(i + ", Error = " + error + ", " + Math.Round(timeSpan.TotalMinutes) + "m (" + Math.Round(timeSpan.TotalSeconds) + "s), eta " + finishTime.ToShortTimeString());
                        epochStart = DateTime.Now;
                    }
                }

                network.Save(@"../../../data/network3.dat");

                TimeSpan runTime = DateTime.Now - startTime;
                Console.WriteLine("Training completed after " + runTime.TotalMinutes + "m.");
                startTime = DateTime.Now;
            }

            // Calculate accuracy.
            double accuracy = Utility.ShowProgressFor<double>(() => Accuracy.CalculateAccuracy(network, inputs, outputs), "Calculating Accuracy");
            Console.WriteLine("Accuracy: " + Math.Round(accuracy * 100, 2) + "%");

            return network;
        }

        /// <summary>
        /// Core machine learning method for parsing csv data, training the svm, and calculating the accuracy.
        /// </summary>
        /// <param name="path">string - path to csv file (training, csv, test).</param>
        /// <param name="count">int - max number of rows to process. This is useful for preparing learning curves, by using gradually increasing values. Use 0 to read all rows.</param>
        /// <param name="machine">MulticlassSupportVectorMachine - Leave null for initial training.</param>
        /// <returns>MulticlassSupportVectorMachine</returns>
        private static MulticlassSupportVectorMachine RunSvm(string path, int count, MulticlassSupportVectorMachine machine = null)
        {
            double[][] inputs;
            int[] outputs;

            // Parse the csv file to get inputs and outputs.
            ReadData(path, count, out inputs, out outputs, new EndStringEndLabelParser());

            if (machine == null)
            {
                // Training.
                MulticlassSupportVectorLearning teacher = null;

                // Create the svm.
                machine = new MulticlassSupportVectorMachine(_pixelCount, new Gaussian(_sigma), _classCount);
                teacher = new MulticlassSupportVectorLearning(machine, inputs, outputs);
                teacher.Algorithm = (svm, classInputs, classOutputs, i, j) => new SequentialMinimalOptimization(svm, classInputs, classOutputs) { CacheSize = 0 };

                // Train the svm.
                Utility.ShowProgressFor(() => teacher.Run(), "Training");
            }

            // Calculate accuracy.
            double accuracy = Utility.ShowProgressFor<double>(() => Accuracy.CalculateAccuracy(machine, inputs, outputs), "Calculating Accuracy");
            Console.WriteLine("Accuracy: " + Math.Round(accuracy * 100, 2) + "%");

            return machine;
        }

        private static int ReadData(string path, int count, out double[][] inputs, out int[] outputs, IRowParser rowParser)
        {
            Parser parser = new Parser(rowParser);

            // Read the training data CSV file and get a resulting array of doubles and output labels.
            List<MLData> rows = Utility.ShowProgressFor<List<MLData>>(() => parser.Parse(path, count), "Reading data");

            // Convert the rows into arrays for processing.
            inputs = rows.Select(t => t.Data.ToArray()).ToArray();
            outputs = rows.Select(t => t.Label).ToArray();

            Console.WriteLine(rows.Count + " rows processed.");

            return rows.Count;
        }
    }
}
