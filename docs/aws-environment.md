# Introduction

This document details the AWS AMI I developed for this project.

### Hardware Configuration

Given my data lives primarily in an S3 bucket, it is tempting to move data in batches to the local store and not bother with EBS backed instances. However, local stores take 5x to 10x longer to boot up and it appears most GPU instances only support EBS. 

### Software Components and Libraries

The image builds upon the DSI-Template3 which was developed for the Galvanize data science immersive as a general purpose environment with Python 2.7 and 3.6 running on Ubuntu 16.03. There are two Deep Learning AMI's created for the program but instructors cautioned these images may not be stable and need work. I am electing to build my own Deep Learning AMI.

- Anaconda 3.6

### Testing

The image is tested with the following scripts.

- src/keras-cifar10-cnn-eval.py
