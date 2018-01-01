# Introduction

I am developing a deep learning neural network to take an image with multiple objects and classify each of the objects.

- Input Data: jpg images of varying size and json documents describing the objects in the image  
    - Convolutional Neural Networks are generally the state of the art for image classification tasks. Newer architectures have emerged that combine CNN's with RNN's to classify multiple objects within one image.
    - Will begin with CNN to start.
- Output: A description of each item in the image (and count of total objects)
    - I can know the total number of classes by extracting the object descriptions from the supplied json files.
    - Listing [extract-classes-from-data.py](../src/extract-classes-from-data.py) extracts this information and builds a dictionary of the unique items.
    - My first run of the script above analyzed more than 536,400 json files and discovered 460,515 unique items. The resultant dictionary has been pickled and is more than 9.6MB.
- Number of Layers:
    - ?
- Parameter (neuron) count per layer
- Weight Initialization Strategy:
- Activation Functions:
- Loss Function:
- Optimization Algorithm:
- Batch Strategy:
- Regularization:


# Tools  

Initially I am starting with the Keras Deep Learning library with the TensorFlow backend and will train on the cifar-10 image dataset to evaluate capabilities and learn the nuances of the framework.

# CIFAR-10 image dataset with Keras and TensorFlow

- [CIFAR-10 Homepage](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/deep_cnn)

###
