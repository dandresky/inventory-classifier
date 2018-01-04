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
