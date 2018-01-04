'''
This script is written to facilitate exploration of the images and associated
json files. The purpose is to gain insights into the data to guide decisions
on model development. Some functions may have value later in the project but
that is not the main goal.
'''
from collections import Counter
from collections import defaultdict
import datetime as dt
import json
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys

IMAGE_DATA_PATH = '../data/bin-images/'
JSON_DATA_PATH = '../data/metadata/'

def get_classes(json_file_list):
    '''
    Return a pair of dictionaries containing statistics for number of classes
    and item distribution in bins.

    item_dict:  class description (key) and the count of the number of times
    that class appears in the list of json files.

    bin_cnt_dict: qty of items in bin (key) and number of bins with that qty
    '''
    item_dict = defaultdict(int)
    bin_cnt_dict = defaultdict(int)
    start_time = dt.datetime.now()
    for i in range(len(json_file_list)):
        with open(JSON_DATA_PATH + json_file_list[i]) as f:
            json_data = json.load(f)
            bin_cnt_dict[json_data['EXPECTED_QUANTITY']] += 1
            for item in json_data['BIN_FCSKU_DATA']:
                item_dict[json_data['BIN_FCSKU_DATA'][item]['asin']] += 1
    stop_time = dt.datetime.now()
    print("Elapsed time = ", (stop_time - start_time).total_seconds(), "s.")
    return item_dict, bin_cnt_dict

def get_image_file_names():
    '''
    Return a list of image file names.
    '''
    file_list = [f for f in listdir(IMAGE_DATA_PATH) if isfile(join(IMAGE_DATA_PATH, f))]
    file_list.sort()
    return file_list

def get_json_file_names():
    '''
    Return a list of json file names.
    '''
    file_list = [f for f in listdir(JSON_DATA_PATH) if isfile(join(JSON_DATA_PATH, f))]
    file_list.sort()
    return file_list

def main():
    # lets first get the number of files and file names in each folder
    image_file_list = get_image_file_names()
    json_file_list = get_json_file_names()
    print("First 10 images: \n", image_file_list[:10])
    print("First 10 json docs: \n", json_file_list[:10])

    if sys.argv[1] == 'getclasses':
        item_dict, bin_cnt_dict = get_classes(json_file_list)

        inventory_counts = Counter(item_dict.values())
        # print(class_counts)
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        ax.bar(list(inventory_counts.keys()), inventory_counts.values(), width=0.4, color='g')
        ax.set_xlim(0, 20)
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        ax.set_ylabel('Number of Items')
        ax.set_xlabel('Inventory Count')
        ax.set_title('Items vs Inventory Counts')
        plt.savefig('item_cnts.png')
        plt.show()

        fig, ax = plt.subplots(1,1, figsize=(8,4))
        ax.bar(list(bin_cnt_dict.keys()), bin_cnt_dict.values(), width=0.4, color='g')
        ax.set_xlim(0, 20)
        ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        ax.set_ylabel('Bins')
        ax.set_xlabel('Items per Bin')
        ax.set_title('Distribution of Items')
        plt.savefig('bin_cnts.png')
        plt.show()
    pass


if __name__ == '__main__':
    main()
