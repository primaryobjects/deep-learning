Deep-Learning Neural Networks with Accord .NET
==============================================

A simple example of using the Accord .NET C# library to implement a deep-learning neural network (ie., Deep Belief Network) with machine learning.

Checkout branch "XOR" for a simple example of deep-learning with Accord .NET. This branch contains training using one of the most basic neural network cases - the XOR function.

Checkout the master branch for a slightly less-basic example of training on an ASCII digit dataset. This example uses multiple layers in the neural network and has the potential to "dream" representations of data within its layers.

Deep-Learning Strategy
----------------------

1. Start with a neural network with multiple RestrictedBoltzman machine layers.
2. Use unsupervised training on each layer in the network, one at a time, except for the output layer. This allows each layer to learn specific features about the input data.
3. If you ran unsupervised training on the whole network, including the output layer, add an additional (untrained) layer to the network to serve as the output layer. Otherwise, skip this step.
4. Run back-propagation on the entire network to fine-tune for classification.
