'''
This script extracts data for all items in the json files. It builds a
dictionary of unique items and their counts in the dataset, as well as item
counts for each bin image.
Initial purpose is to obtain statistics for the dataset.
'''
import boto3
from collections import Counter
from collections import defaultdict
import datetime as dt
import json
import matplotlib.pyplot as plt
import pickle
from statistics import mode
import sys

# constants and defines
#BATCH_SIZE = 1000
BUCKET_NAME = 'dea-inventory-classifier'

def get_classes():
    # set up the boto3 connection to s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    s3.meta.client.head_bucket(Bucket=BUCKET_NAME)

    # loop over all keys in the metadata folder and process all json files
    item_dict = defaultdict(int)
    bin_cnt_dict = defaultdict(int)
    cnt = 0
    start_time = dt.datetime.now()
    for key in bucket.objects.filter(Prefix='metadata/'):
        # Interested only in json files (filter returns path)
        if key.key.endswith('.json'):
            item_dict, bin_cnt_dict = process_file(s3, key.key, item_dict, bin_cnt_dict)
            cnt += 1
            if cnt % 100 == 0:
                print("Processed ", cnt, " out of 500k+ files.")
    stop_time = dt.datetime.now()
    print("Elapsed time = ", (stop_time - start_time).total_seconds(), "s.")
    return item_dict, bin_cnt_dict


def process_file(s3, file_path, item_dict, bin_cnt_dict):
    #item_dict = defaultdict(int)
    obj = s3.Object(BUCKET_NAME, file_path)
    json_str = obj.get()['Body'].read().decode('utf-8') # convert to string
    # for each item in a json file, extract the 'asin' and 'name' data and add
    # to the item dictionary.
    json_data = json.loads(json_str)
    bin_cnt_dict[json_data['EXPECTED_QUANTITY']] += 1
    for item in json_data['BIN_FCSKU_DATA']:
        item_dict[json_data['BIN_FCSKU_DATA'][item]['asin']] += 1
    return item_dict, bin_cnt_dict


if __name__ == '__main__':
    if sys.argv[1] == 'getclasses':
        # read all of the json docs and get a count of the unique classes
        # This takes 16hrs on PC and 5hrs on lowest level GPU
        classes, bin_cnts = get_classes()
        print("There are ", len(classes.keys()), "classes.")

        with open('pkl_classes.pickle', 'wb') as handle:
            pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('pkl_bin_cnts.pickle', 'wb') as handle:
            pickle.dump(bin_cnts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif sys.argv[1] == 'plotclasses':
        # read the previously pickled dictionary and plot the frequency of items
        classes = {}
        with open('pkl_classes.pickle', 'rb') as handle:
            classes = pickle.load(handle)
        inventory_counts = Counter(classes.values())
        # print(class_counts)
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        bars = ax.bar(list(inventory_counts.keys()), inventory_counts.values(), width=0.4, color='g')
        ax.set_xlim(0, 20)
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        ax.set_ylabel('Number of Items')
        ax.set_xlabel('Inventory Count')
        ax.set_title('Items vs Inventory Counts')
        plt.savefig('item_cnts.png')
        plt.show()

    elif sys.argv[1] == 'other':

        with open('pkl_classes.pickle', 'rb') as handle:
            classes = pickle.load(handle)
            print("Total number of items = ", sum(classes.values()))

        with open('pkl_bin_cnts.pickle', 'rb') as handle:
            bin_cnts = pickle.load(handle)
            print("Total bin counts = ", len(bin_cnts))
            fig, ax = plt.subplots(1,1, figsize=(8,4))
            bars = ax.bar(list(bin_cnts.keys()), bin_cnts.values(), width=0.4, color='g')
            ax.set_xlim(0, 20)
            ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            ax.set_ylabel('Bins')
            ax.set_xlabel('Items per Bin')
            ax.set_title('Distribution of Items')
            plt.savefig('bin_cnts.png')
            plt.show()

    else:
        print("Argument required (getclasses, plotclasses)")
