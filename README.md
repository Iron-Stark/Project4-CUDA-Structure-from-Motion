CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Dewang Sultania
  * [LinkedIn](https://www.linkedin.com/in/dewang-sultania/)
* Tested on: Windows 10, Intel Xeon E-2176M @ 2.70GHz 16GB, Quadro P2000 4GB (Personal Computer)
* Dependencies: cublas, opencv, curand

### Table of Contents
1. [Overview](#overview)
2.   [Multilayer Perceptron](#mlp)
3.   [Architecture](#architecture)
4.   [Workflow Diagram](#workflow)
5.   [Performance Analysis](#performance)
6.   [Extending it to MNIST Dataset](#mnist)
7.   [Future Work](#future)

<a name = "overview"/>

## Overview

This repository contains code for a fully connected neural network in CUDA. Includes forward propagation, back propagation and kernels for activation functions. The network is trained to identify characters of the english alphabet. It was then later extended to train on the MNIST dataset as well.

<a name = "mlp"/>

## Multilayer Perceptron

Neural Network             |  Neuron
:-------------------------:|:-------------------------:
![](img/neural_network.JPG) | ![](img/neuron.JPG)


Neural Networks are modeled as collections of neurons that are connected in an acyclic graph. In other words, the outputs of some neurons can become inputs to other neurons. Cycles are not allowed since that would imply an infinite loop in the forward pass of a network. Instead of an amorphous blobs of connected neurons, Neural Network models are often organized into distinct layers of neurons. For regular neural networks, the most common layer type is the fully-connected layer in which neurons between two adjacent layers are fully pairwise connected, but neurons within a single layer share no connections.

A multilayer perceptron (MLP) is a class of feedforward artificial neural network. An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

##### Terminology

There are certain terms that are widely used when describing neural networks. I will try to give a concise definition of the commonly used terms throughout the readme here.

- **Input Layer**:  The input layer is basically the inputs to the neural network, the dimensions of the input layer is same as the dimension of the input.

- **Hidden Layer**: A hidden layer in an artificial neural network is a layer in between input layers and output layers, where artificial neurons take in a set of weighted inputs and produce an output through an activation function

- **Output Layer**:  The output layer in an artificial neural network is the last layer of neurons that produces given outputs for the program.

- **Activation Functions**:  Every activation function (or non-linearity) takes a single number and performs a certain fixed mathematical operation on it. The non-linearity is critical computationally - if we left it out, the two matrices could be collapsed to a single matrix, and therefore the predicted class scores would again be a linear function of the input. The non-linearity is where we get the wiggle.

You can learn more about them [here](http://cs231n.github.io)

<a name = "architecture"/>

## Architecture:

The network contains 196 dimensions in the input layer, 100 in the first hidden layer, 50 in the second, 25 in the third and finally 52 outputs for 52 classes. The activation function used in the hidden layers was ReLU activation and the final layer uses softmax. Softmax is used because it predicts the output as probability of belongingness to each class.

There are certain advantages to using ReLU over its counterparts like sigmoid and tanh. Namely:

- It was found to greatly accelerate (e.g. a factor of 6 in [Krizhevsky et al.](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)) the convergence of stochastic gradient descent compared to the sigmoid/tanh functions. It is argued that this is due to its linear, non-saturating form.
- Compared to tanh/sigmoid neurons that involve expensive operations (exponentials, etc.), the ReLU can be implemented by simply thresholding a matrix of activations at zero.

**Weight Initialization**: In case of neural networks, it is critical to start at the right place because the loss function is not convex, hence gradient descent is not guaranteed to find the optimal solution. Since the neural network has ReLU activation function, I used He-normal initialization for the starting point of weights. More details on this can be found [here](https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528)

The forward and the backward propagation equations are as shown in the figure.
 ![](img/for_back.JPG)

The update equations are:
$$
w^{[l]} = w^{[l]} - \alpha dw^{[l]}\\
b^{[l]} = b^{[l]} - \alpha db^{[l]}\\
$$

The matrix multiplication steps here were done using cuBlas:sgemm() API, which was not an easy task to interpret when it came to transpose multiplications. Basically the parameters lda, ldb, ldc were not properly documented in the documentation and took a lot of time to figure out.

Alpha here is the learning rate, Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. The lower the value, the slower we travel along the downward slope. While this might be a good idea (using a low learning rate) in terms of making sure that we do not miss any local minima, it could also mean that we’ll be taking a long time to converge — especially if we get stuck on a plateau region.


Optimal Learning Rate             |  Learning Rate affects
:-------------------------:|:-------------------------:
![](img/learning_rate.png) | ![](img/lr.png)

Doing this over and over again for 1000 epochs over the whole training set gets 100% accuracy.

<a name = "workflow"/>

## Workflow Diagram

![](img/workflow.png)

<a name = "performance"/>

## Performance Analysis

I trained the network for 100000 iterations on the given data, while randomly sampling from it with a learning rate of 0.0005. The loss curve for the network is:

![](img/loss_charac.png)

Accuracy:  100%

<a name = "mnist"/>

##  Extending the network to MNIST dataset (Extra Credit)

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. 

The way I had written my code made it easy to extend to other datasets which can be read as opencv Mat images, so I extended the network to train on the MNIST dataset. I only had to add a few lines of code to the main.cpp function to read in the dataset and no changes were required in any other file. To toggle training between mnist and character recognition, just turn the mnist flag on line number 18 of main.cpp to 1/0

The loss curves mnist dataset:

![](img/loss_mnist.png)


Training Accuracy: 96.48%
Testing Accuracy: 95.65%

<a name = "future"/>

## Future Work

Deep Learning is highly sensitive to and has lots of hyperparameters, starting from architecture, learning rate, etc. There are a lot of other things to try out as well like learning rate decay, optimization techniques like adam and momentum, regularization techniques like weight decay, lasso, batchnorm, dropout, etc. There was an overwhelming amount of choices that can be made here. The only way to know what will work and what won't is to try these out. I have played around with all those choices a lot in Pytorch and Tensorflow, but will keep them out of scope for this project.

#### References
[1 https://www.deciphertechnic.com/install-opencv-with-visual-studio/](https://www.deciphertechnic.com/install-opencv-with-visual-studio/)
[2 http://cs231n.github.io](http://cs231n.github.io)
[3 http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

