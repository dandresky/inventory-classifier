# Inventory Classifier - Material count and classification using Convolutional Neural Net

by David Andresky

As an electrical engineer I have developed numerous electronic devices and partnered with operations on the introduction and ramp up of my products. From domestic fulfillment centers to large contract manufacturing facilities in Asia, there are consistent pain points and inefficiencies with regards to material handling and inventory management. Some common issues include:

- Selecting the right component
- Warehouse space utilization
- Just in time throughput
- Inefficient distribution of materials
- Lack of automation

Machine learning tools can be leveraged to help solve these crucial business issues and many more. With this project, I intend to design a deep learning neural network capable of counting and identifying inventory items in a warehouse from images of their storage locations. Armed with this capability, robotic tools and systems can be developed to automate processes allowing material and production planners to improve operations efficiency and reduce costs.

### Minimum Viable Product

Due to the time frames for the capstone project, the MVP will be a trained network that can count each of the distinct items in a storage location. As time permits, I will expand the project to classify each item and ultimately identify with unique part numbers/descriptions.

# The Data

The neural network wil be developed and trained on the Amazon Bin Image dataset.

# Project Plan

The Inventory Classifier will be developed and trained using the Amazon Bin Image Data Set.

Amazon has an automated inventory management system where items are stored in random locations based on space available. As part of their open data program, they have made more than half a million images of inventory bins from one of their fulfillment centers available for research, along with JSON documents describing the contents of each of the bins that can be used to train the model on.  

The image and json document below are an example of what has been provided.  

![](img/523.jpg)

![Associated JSON file](data/523.json)

My notes on the implementation details can be found in the following documents:
- [Amazon Machine Image (AWS)](docs/aws-environment) - Setup details for the AMI I created for this project.
- [Data Processing](docs/data-processing) - Notes on the images and JSON files supplied and how AWS S3 and EC2 are leveraged for storage and processing.
- [Model Development](docs/model-development.md) - Notes on my strategy for developing the neural net model architecture. Includes intermediate steps or tests that may not make it in the final model.

# Source Files

The following is a brief overview of the source files and their use in the project.  

- [extract-classes-from-data.py](src/extract-classes-from-data.py) - a script that builds a dictionary of unique classes. Initial purpose is to know how many outputs are required for the neural net.
- [keras-cifar10-cnn-eval.py](src/keras-cifar10-cnn-eval.py) - a script with an example CNN architecture that classifies images from the CIFAR-10 dataset. Purpose is to test EC2 AMI's with an architecture that has a known baseline and to explore some of the Keras framework.
- [plot-test.py](src/plot-test.py) - a script to test setup of visualization code used in the model scripts. The model scripts run for many hours and this file is just used as a testbed.
