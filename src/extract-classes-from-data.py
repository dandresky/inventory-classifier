'''
This script extracts the 'asin' and 'name' data for each item in a json file
and builds a dictionary where 'asin' is the key and 'name' is the value.
Initial purpose is to identify all of the classes in the dataset.
'''
import boto3
import botocore
from collections import Counter
from collections import defaultdict
import datetime as dt
import json
import matplotlib.pyplot as plt
import pickle
import urllib
import sys

# constants and defines
BATCH_SIZE = 1000
BUCKET_NAME = 'dea-inventory-classifier'

def get_classes():
    # set up the boto3 connection to s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    s3.meta.client.head_bucket(Bucket=BUCKET_NAME)

    # loop over all keys in the metadata folder and process all json files
    item_dict = defaultdict(int)
    cnt = 0
    start_time = dt.datetime.now()
    for key in bucket.objects.filter(Prefix='metadata/'):
        # Interested only in json files (filter returns path)
        if key.key.endswith('.json'):
            item_dict = process_file(s3, key.key, item_dict)
            cnt += 1
            if cnt % 100 == 0:
                print("Processed ", cnt, " out of 500k+ files.")
    stop_time = dt.datetime.now()
    print("Elapsed time = ", (stop_time - start_time).total_seconds(), "s.")
    return item_dict

def process_file(s3, file_path, item_dict):
    #item_dict = defaultdict(int)
    obj = s3.Object(BUCKET_NAME, file_path)
    json_str = obj.get()['Body'].read().decode('utf-8') # convert to string
    # for each item in a json file, extract the 'asin' and 'name' data and add
    # to the item dictionary.
    json_data = json.loads(json_str)
    for item in json_data['BIN_FCSKU_DATA']:
        item_dict[json_data['BIN_FCSKU_DATA'][item]['asin']] += 1
        # item_dict.update({item: json_data['BIN_FCSKU_DATA'][item]['name']})
    return item_dict


if __name__ == '__main__':
    if sys.argv[1] == 'getclasses':
        # read all of the json docs and get a count of the unique classes
        # This takes 16hrs on PC and 5hrs on lowest level GPU
        classes = get_classes()
        print("There are ", len(classes.keys()), "classes.")
        with open('item_dict.pickle', 'wb') as handle:
            pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('item_dict.pickle', 'rb') as handle:
            read_back = pickle.load(handle)
            print(read_back == classes)
            print(read_back)
    elif sys.argv[1] == 'plotclasses':
        # read the previously pickled dictionary and plot the frequency of items
        classes = {}
        with open('item_dict.pickle', 'rb') as handle:
            classes = pickle.load(handle)
        class_counts = Counter(classes.values())
        # print(class_counts)
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        bars = ax.bar(list(class_counts.keys()), class_counts.values(), width=0.4, color='g')
        ax.set_xlim(0, 20)
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        ax.set_ylabel('Number of Items')
        ax.set_xlabel('Inventory Count')
        ax.set_title('Items vs Inventory Counts')
        plt.savefig('item_cnts.png')
        plt.show()

    else:
        print("Argument required (getclasses, plotclasses)")
