# Introduction

This document details the AWS AMI I developed for this project.

### Hardware Configuration

All data is obtained and stored in a S3 bucket for low-cost long-term storage. When the project is completed the S3 bucket can be moved to Glacier storage.

A 200GB EBS volume is provisioned from an [Amazon Deep Learning Image (AMI)](https://aws.amazon.com/machine-learning/amis/) that includes most of the current deep learning tools such as Tensorflow, Theano, Keras, Anaconda, etc. that can be accessed through various canned environments. This Image will be updated as needed to execute the project and the volume can be attached to a variety of multi-core GPU instances.

Note: [Instructions on attach/detach of volumes and changing size](https://n2ws.com/how-to-guides/how-to-increase-the-size-of-an-aws-ebs-cloud-volume-attached-to-a-linux-machine.html)

All data will be copied to the EBS volume for fast and frequent access. When the project is complete, the image data will be deleted from the EBS volume, all new data generated such as models, reports, etc. will be moved to S3, a final Image will be created and saved to S3 and then the EC2 instance and EBS volumes will be deleted.

### Software Components and Libraries

The image builds upon the [Amazon Deep Learning Image](https://aws.amazon.com/machine-learning/amis/) that includes most of the current deep learning tools such as Tensorflow, Theano, Keras, Anaconda, etc. The following is a list of software components I have added to this baseline and the environments that use it.

| Software Component    | Description / Environments                |
| --------------------- | ----------------------------------------- |
| pillow                | Image processing, All environments        |
| python-resize-image   | Image processing, All environments        |
