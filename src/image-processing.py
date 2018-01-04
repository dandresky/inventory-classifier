'''
'''
import boto3
import datetime as dt
from PIL import Image
import resizeimage as ri
import sys

# constants and defines
BUCKET_NAME = 'dea-inventory-classifier'

def change_image_size(im):
    pass

def get_image_sizes():
    # this function is written to scan all images in the S3 bucket and return
    # a dictionary of sizes. The key is the file name, the value is a tuple
    # of width and height.
    # set up the boto3 connection to s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    s3.meta.client.head_bucket(Bucket=BUCKET_NAME)

    cnt = 0
    start_time = dt.datetime.now()
    for key in bucket.objects.filter(Prefix='bin-images/'):
        # Interested only in json files (filter returns path)
        if key.key.endswith('.jpg'):
            full_path = "http://s3.amazonaws.com/" + BUCKET_NAME + "/" + key.key
            print(full_path)
            width, height = get_image_size(full_path)
            print("Width = ", width, ", height = ", height)
            cnt += 1
            if cnt % 100 == 0:
                print("Processed ", cnt, " out of 500k+ files.")
                break
    stop_time = dt.datetime.now()
    print("Elapsed time = ", (stop_time - start_time).total_seconds(), "s.")

def get_image_size(im):
    with open(im, 'r+b') as f:
        with Image.open(f) as image:
            return image.size

if __name__ == '__main__':
    # test functions and query files for info
    if sys.argv[1] == 'test':
        # the 2nd argument will be a image and its path
        # Get the size of the image
        get_image_sizes()
    elif sys.argv[1] == 'one':
        width, height = get_image_size(sys.argv[2])
        print(width)
        print(height)
