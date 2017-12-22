My random notes on Neural Net development tools and libraries considered for this project.

# High Level Deep Learning Frameworks

### [Keras](https://keras.io/)

From website:

"Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation."

Keras offers a gentle introduction into development and tuning of neural nets, supporting both CNN and RNN architectures. It supports TensorFlow, Theano, or CNTK as its backend and is well documented.

### [Deep Learning 4J](https://deeplearning4j.org/)

### [Caffe](http://caffe.berkeleyvision.org/)

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR) and by community contributors.


# Low Level Deep Learning Frameworks

Supported by Keras and DL4J
- TensorFlow
- Theano
- CNTK

[This blog](http://minimaxir.com/2017/06/keras-cntk/) documents some interesting benchmark tests between TensorFlow and CNTK on Keras. It suggests that TensorFlow would be the best framework for CNN's based on both speed and accuracy. Multiple other sites found in a Google search consistently rated Theano as slower than both TensorFlow and CNTK. Lastly, Keras recommends TensorFlow as the preferred backend.

### TensorFlow

From website:

"TensorFlow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well."  

TensorFlow is supported by Keras as its backend for vector, matrix, and tensor operations used in neural net development.

Tensorflow ...
- API has several levels
    - TensorFlow Core - for researchers and others who require a fine level of control over their models.
    - Multiple high level API's that make TensorFlow easier to learn.
    - programmed operations and connections can be graphically rendered using TensorBoard
- If I am ambitious I could build my own architecture on top of the API

### Theano

Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. It is a Python library that claims to rival C implementations of its mathematical operations, especially on GPU's.  

Theono is supported by Keras as its backend for vector, matrix, and tensor operations used in neural net development.

### CNTK - Microsoft Cognitive Toolkit
