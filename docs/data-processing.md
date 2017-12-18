# Introduction

Amazon has an automated inventory management system where items are stored in random locations based on space available. As part of their open data program, they have made approximately half a million images of inventory bins from one of their fulfillment centers available for research, along with JSON documents describing the contents of each of the bins that can be used to train the model on.  

The image (27.8kB) and json document (1.3kB) below are an example of what has been provided.  

![](../img/523.jpg)

[Associated JSON file](../data/523.json)

# Memory/Storage Needs

If all images are assumed to be 30kB and all json docs are assumed to be 2kB, then 500k images and docs would require approximately 16GB of storage. 25GB to 30GB should provide plenty of margin and allow for a second source of training data should I find one.  

For development and evaluation of models I probably only require 100 to 1000 records which can be stored on my local machine where development will take place. Training will occur on an AWS EC2 instance where I expect to implement a mini-batch training process, perhaps parallelized over multiple cores. During training I would pull anywhere from 1000 to 10000 records at a time into the local data store but it does not need to persist after training.

The process described above means an S3 bucket will be sufficient to persistently store all of my training and validation data. I can avoid the need for EBS, saving me money and allowing me to easily experiment with different EC2 instance types for optimal performance.

Lastly, S3 storage can easily be moved to Glacier storage when I am done so that I can persist all of my data when I am done for for only $0.10 a month.
