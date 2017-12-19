# Introduction

Amazon has an automated inventory management system where items are stored in random locations based on space available. As part of their open data program, they have made approximately half a million images of inventory bins from one of their fulfillment centers available for research, along with JSON documents describing the contents of each of the bins that can be used to train the model on.  

The image (27.8kB) and json document (1.3kB) below are an example of what has been provided.  

![](../img/523.jpg)

[Associated JSON file](../data/523.json)

To obtain list of file names and file sizes requires me to use the s3 API to list objects and write metadata to a file. The following command yields a file that is nearly 500MB. An example of the info for each record is given below.

$aws s3api list-objects --bucket aft-vbi-pds --output json >> bucket-metadata.json

{  
- "Size": 72867,  
- "Owner": {  
    - "ID":   "8f1e6b95f476ad59204b71fc0a699779fe5be4d36bca43c5b27e5ee5ba6c8817",  
    - "DisplayName": "aft-cv-dataset-release"  
    },  
- "LastModified": "2017-01-13T22:47:54.000Z",  
- "StorageClass": "STANDARD",  
- "Key": "bin-images/00025.jpg",  
- "ETag": "\"0f54e37796ff83eda4e6935f4a0d9312\""  

},  

To reduce the size, I will limit request to retrieve just the file name (Key) and the size (Size) as follows:

aws s3api list-objects --bucket aft-vbi-pds --output json --query 'Contents[].{Key: Key, Size: Size}' >> bucket-contents.json

This command yields a file size of approximately 79MB.

Some statistics (see [Jupyter Notebook](../data/bucket-analysis.ipynb)):
- There are 536434 images and associated json files
- Total size of all files: 31.6GB
- Total size of images: 30.5GB (mean = 56.8kB)
- Total size of metadata: 1.1GB (mean = 2.05kB)


# Memory/Storage Needs

After reading the metadata for all of the files in the bucket, I compute just a little over 31GB of data exists. If I allow for a second source of training data (should I find one), then at least 50G of storage should be planned for. I may also need to do a transformation on the images to get them all to a common size (num of pixels). In that event, I will only store the modified images in my repository, and not copy the original images.  

For development and evaluation of models I probably only require 100 to 1000 records which can be stored on my local machine where development will take place. Training will occur on an AWS EC2 instance where I expect to implement a mini-batch training process, perhaps parallelized over multiple cores. During training I would pull anywhere from 1000 to 10000 records at a time into the local data store but it does not need to persist after training.

The process described above means an S3 bucket will be sufficient to persistently store all of my training and validation data. I can avoid the need for EBS, saving me money and allowing me to easily experiment with different EC2 instance types for optimal performance.

Lastly, S3 storage can easily be moved to Glacier storage when I am done so that I can persist all of my data when I am done for for only $0.10 a month.
