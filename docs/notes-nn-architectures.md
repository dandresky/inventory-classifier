# Overview

Core Components  
- Parameters (weights?)
- Layers (neurons and connections)
- Activation functions (consistent across layer)
- Loss functions
- Optimization methods (Gradient Descent)
- Hyperparameters

Building Blocks for larger deep learning networks (beyond basic preceptrons and MLN's)  
- Feed-forward multi-layer networks
- RBM's - Restricted Boltzman Machines
- Autoencoders

Architectures
- UPN's - Unsupervised Pretrained Networks
- CNN's - Convolutional Neural Network
- RNN's - Recurrent Neural Network
- Recursive Neural Network
- Recurrent Models of Visualization (Hybrid CNN and RNN)

NOTE: CNN's are likely the best architecture for my project. However, a hybrid architecture exists called Recurrent Models of Visual Attention (see p.257 of Deep Learning book) that may be useful. These models are effective at dealing with images that are cluttered with multiple objects (as the Amazon images will be) that are difficult for traditional CNN's to deal with. It blends CNN for raw perception and RNN for time-domain. There is a second hybrid described as well that is designed to classify objects in a cluttered image.

NOTE: Recursive Neural Net or some hybrid may ultimately be the best choice.

# Activation Functions  

- Linear
    - typically used on input layers, also preferred for output layers with single real-valued number
- Sigmoid
    - squashes output to values between 0 and 1
    - have fallen out of favor in recent years and replaced with RELU or TANH
    - appropriate for outputs with a single binary classification interpreted as a probability distribution or to make multiple classifications.
- TANH
    - can deal with negative numbers
- Hard TANH
- Softmax
    - can be applied to continuous data
    - typically used on output layers where we are selecting one class from multiple classes with best probability distribution
- RELU - Rectified Linear
    - recently replacing Sigmoid and TANH as the preferred activation function as it does not suffer from vanishing gradient issues
- Leaky RELU
    - negative numbers are given a small slope instead of forced to 0
- Softplus
    - smoothed version of RELU with a curve at 0 instead of a hard change.

# Loss Functions

We use loss functions to determine the penalty for incorrect classification.  

For regression
- MSE - Mean Squared Error
- MAE - Mean Absolute Error
- MSLE - Mean Squared Log Error
- MAPE - Mean Absolute Percentage Error

For Classification
- Hinge Loss
- Logistic Loss
- Negative Log Likelihood
- Squared Loss

For Reconstruction (unsupervised feature extraction)
- Cross-Entropy Loss
    - apply Gaussian noise and then the loss function punishes the network for any result that is less similar to the original input
    - Drives network to learn different features
    - used for feature engineering in the pretrain phase that involves RBM's

# Hyperparameters

- Activations
- Layer Size
- Learning rate
    - coefficient that scales size of learning steps
    - too fast can overshoot minimum and oscillate, too slow is computationally intensive
    - we can schedule learning rates to decrease over time to allow large steps initially while avoiding overshoot
- Regularization
    - controls parameter size over time and is used to combat overfitting
    - methods:
        - Dropout - randomly drops neurons - mutes parts of the input to each layer such that the network learns other portions
        - Dropconnect - drops connections to neurons
        - L1 penalty - prevents parameter space from getting too big in any direction
        - L2 Penalty - computationally more efficient but does not do feature selection
- Momentum
    - helps algorithm get out of spots in the error space where it might become stuck
    - regulates learning rate? Is this the scheduling process referred to in learning rate?
    - Methods:
        - Nesterovs's Momentum
        - Adagrad
        - RMSProp
        - Adadelta
        - ADAM
- Sparsity
    - Sparsity is the idea that some features that need to be learned will be rare in a data set - limited number of nodes will activate impeding learning.
    - biases can help by forcing some number of nodes to activate, improving ability to learn
- Weight Initialization Strategy
- Selection of loss function
- Settings for epochs (mini-batch size)
- Normalization scheme for input data

# Optimization

First order optimization algorithms
- calculate the Jacobian matrix which is the partial derivative of the loss function
- one derivative per parameter, finds best path to a minimum
- Gradient descent is a type of 1st order algorithm
- Stochastic Gradient Descent trains faster than batch, min-batch is also fast

Second order optimization algorithms
- derivative of the Jacobian matrix (second order derivative of the loss function)
- takes into account interdependencies between parameters when choosing how much to modify each parameter
- takes better steps (converges faster) but adds time to computations
- method types:
    - Limited memory BFGS
    - Conjugate Gradient
    - Hessian free

Other algorithms
- Genetic
- Particle swarm
- Ant colony
- Simulated annealing

# Architecture Building Blocks

### RBM - Restricted Boltzman machine

- models probability and great at feature extraction and dimensionality reduction
- used for pretraining layers in larger deep networks
- 5 main components: Visible nodes, hidden nodes (feature detector), weights, visible bias units, hidden bias units
- A standard RBM has a visible layer and a hidden layer
- Every visible unit is connected to every hidden unit but connections between nodes of the same layer are prohibited
- Each layer contains a bias node that is always on and connected to every node in the layer
- Used in reconstruction: feature engineering from unlabeled data - the weights learned are used to initialize weights in larger networks

### Autoencoder

- unsupervised feature extraction because only uses original input data to learn weights rather than back propagation
- used to learn compressed representations of data sets; output is reconstruction of input in most efficient form
- used for unsupervised learning of unlabeled data
- output layer has same number of nodes and input
- designed to reproduce its own input data (as breakdown of features? i.e. input is a face, output is nose, mouth, eyes, etc.?), this makes it a good anomaly detector
- given quality of Amazon images, could a denoising autoencoder be useful?


# Architectures

- UPN's - Unsupervised Pretrained Networks
    - Autoencoders
    - DBN - Deep Belief Network
    - GAN - Generative Adversarial Network
- Convolutional Neural Network

### DBN - Deep Belief network

- Composed of one or more layers of RBM's for pretrain phase in parallel with a feed-forward network for fine tuning
- Fundamental purpose of RBM layers is to learn higher level features from raw input data in an unsupervised manner
- This improves neural network buy letting the RBM learn the higher level features
- WE THEN USE THE RBM LAYER FEATURES AS THE INITIAL WEIGHTS IN THE FEED-FORWARD NETWORK TO HELP GUIDE THE PARAMETERS TOWARD BETTER REGIONS OF THE SEARCH SPACE!
- We can do normal back propagation in the feed-forward network with a low learning rate because pretrain did a general search of the parameter space
- CNN's seem to have taken over as the model of choice for image processing.

### Convolutional Neural Network

- Images have structure that would yield too many weights and features in the normal network - length and width in pixels and depth represented by RGB colors
- CNN's take advantage of this structure to gain computational efficiency
- Layer groups
    - Imput Layer
    - Feature extraction layer
        - Convolution
        - ReLU
        - Pool
    - Classification layer (fully connected)
- Neurons in a convolution layer are only connected to a small region of neurons in the previous layer




# End
