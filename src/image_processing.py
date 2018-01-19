import datetime as dt
import json
from keras.preprocessing.image import img_to_array
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import random
from resizeimage import resizeimage
from sklearn.model_selection import train_test_split
import sys

IMAGE_DATA_PATH = '../../dsi-capstone-data/data/bin-images/'
JSON_DATA_PATH = '../../dsi-capstone-data/data/metadata/'

'''
The ImageProcessing class provides an object that analyses the raw data folders
to determine and store the following information:
    1) total number of image/json files
    2) labels for each image, list of unique labels, and list of missing labels

Functions:

    pre_process_images()
        reads images, converts data to arrays, and resizes to a common size.
'''
class ImageProcessing(object):

    def __init__(self, target_size=(150,150), max_qty=None):
        # self.image_path = image_path
        self.target_size = target_size
        self.max_qty = max_qty
        # get an array of image and json file names, randomly shuffled
        self.image_files, self.json_files = self._get_file_name_arrays()
        # get an array of labels in the same shuffled order as the image and
        # json files
        self.labels = self._extract_labels()
        # missing_labels and unique_labels are only used to understand output
        # layer structure of the neural network
        self.unique_labels = self._get_unique_labels()
        self.missing_labels = self._get_missing_labels()
        pass

    def pre_process_images(self):
        '''
        Pre-process all images and save data as numpy arrays to disk. This is
        alternative method to batch processing in first version. This is called
        once from the console, then the model accesses the saved numpy arrays
        for training and test.
        '''
        # filter out the image files, json files, and labels that exceed the
        # maximum quantity. This is to strip out the the outliers (bin
        # quanities that are too large to detect)
        if self.max_qty:
            mask = np.where(self.labels <= self.max_qty)
            self.image_files = self.image_files[mask]
            self.json_files = self.json_files[mask]
            self.labels = self.labels[mask]

        # create the train test split
        train_img, test_img, train_lbl, test_lbl = \
            train_test_split(self.image_files,
                             self.labels,
                             test_size=0.20,
                             random_state=39)

        # create the processed training image array. Pixel values saved are
        # uint8 to save space. Normalization needs to be done in the model.
        print('\nPre-processing training images ... ...')
        start_time = dt.datetime.now()

        depth = 3
        arr = np.zeros((len(train_img), self.target_size[0], self.target_size[1], depth), dtype=np.uint8)
        for idx, img in enumerate(train_img):
            with open(IMAGE_DATA_PATH + img, 'r+b') as f:
                with Image.open(f) as image:
                    resized_image = resizeimage.resize_contain(image, self.target_size)
                    resized_image = resized_image.convert("RGB")
                    #resized_image.save(IMAGE_DATA_PATH + 'resized-' + self.X_train[self.batch_index], image.format)
                    X = img_to_array(resized_image).astype(np.uint8)
                    arr[idx] = X
            if (idx + 1) % 1000 == 0:
                print(idx+1, "out of", len(train_img), "training images have been processed")
        stop_time = dt.datetime.now()
        print("Pre-processing of training images took ", (stop_time - start_time).total_seconds(), "s.\n")

        print('\nSaving the processed training images array ... ...')
        start_time = dt.datetime.now()

        print("Size of numpy array = ", sys.getsizeof(arr))
        np.save('../../dsi-capstone-data/processed_training_images.npy', arr)

        stop_time = dt.datetime.now()
        print("Saving processed array took ", (stop_time - start_time).total_seconds(), "s.\n")

        # create the processed test image array. Pixel values saved are
        # uint8 to save space. Normalization needs to be done in the model?
        print('\nPre-processing test images ... ...')
        start_time = dt.datetime.now()
        depth = 3
        arr = np.zeros((len(test_img), self.target_size[0], self.target_size[1], depth), dtype=np.uint8)
        for idx, img in enumerate(test_img):
            with open(IMAGE_DATA_PATH + img, 'r+b') as f:
                with Image.open(f) as image:
                    resized_image = resizeimage.resize_contain(image, self.target_size)
                    resized_image = resized_image.convert("RGB")
                    #resized_image.save(IMAGE_DATA_PATH + 'resized-' + self.X_train[self.batch_index], image.format)
                    X = img_to_array(resized_image).astype(np.uint8)
                    arr[idx] = X
            if (idx + 1) % 1000 == 0:
                print(idx+1, "out of", len(test_img), "test images have been processed")
        stop_time = dt.datetime.now()
        print("Pre-processing took ", (stop_time - start_time).total_seconds(), "s.\n")

        print('\nSaving the processed test images array ... ...')
        start_time = dt.datetime.now()

        print("Size of numpy array = ", sys.getsizeof(arr))
        np.save('../../dsi-capstone-data/processed_test_images.npy', arr)

        stop_time = dt.datetime.now()
        print("Saving array took ", (stop_time - start_time).total_seconds(), "s.\n")

        print('\nSaving the train/test label arrays ... ...')
        start_time = dt.datetime.now()

        print("Size of training labels numpy array = ", sys.getsizeof(train_lbl))
        np.save('../../dsi-capstone-data/training_labels.npy', train_lbl)
        print("Size of test labels numpy array = ", sys.getsizeof(test_lbl))
        np.save('../../dsi-capstone-data/test_labels.npy', test_lbl)

        stop_time = dt.datetime.now()
        print("Saving arrays took ", (stop_time - start_time).total_seconds(), "s.\n")

        pass

    '''----------------------------------------------------------------------
    Private functions of the ImageProcessing class
    ----------------------------------------------------------------------'''
    def _extract_labels(self):
        '''
        read json files and extract bin qty for each image. These are the
        labels for each image.
        '''
        print("\nExtracting bin quantity labels for each image ... ...")
        start_time = dt.datetime.now()
        labels = []
        for idx, filename in enumerate(self.json_files):
            with open(JSON_DATA_PATH + filename) as f:
                json_data = json.load(f)
                qty = json_data['EXPECTED_QUANTITY']
                labels.append(qty)
        stop_time = dt.datetime.now()
        print("Extracting took ", (stop_time - start_time).total_seconds(), "s.\n")
        return np.array(labels).astype(np.uint8)

    def _get_file_name_arrays(self):
        '''
        Return arrays of image file names and JSON file names, randomly
        shuffled but maintaining consistent order between the two
        '''
        print("\nScanning and shuffling image and json files and ... ...")
        start_time = dt.datetime.now()
        # get list of all image files and sort by file name
        img_file_list = [f for f in listdir(IMAGE_DATA_PATH) if isfile(join(IMAGE_DATA_PATH, f))]
        img_file_list.sort()
        # get list of all json files and sort by file name
        json_file_list = [f for f in listdir(JSON_DATA_PATH) if isfile(join(JSON_DATA_PATH, f))]
        json_file_list.sort()

        # randomly shuffle the image list and make json list consistent
        new_list = list(zip(img_file_list, json_file_list))
        random.shuffle(new_list)
        img_file_list, json_file_list = zip(*new_list)
        stop_time = dt.datetime.now()
        print("Scanning and shuffling took ", (stop_time - start_time).total_seconds(), "s.\n")
        return np.array(img_file_list), np.array(json_file_list)

    def _get_missing_labels(self):
        '''
        Return the integer quantities missing from the labels
        '''
        start, end = self.unique_labels[0], self.unique_labels[-1]
        return sorted(set(range(start, end + 1)).difference(self.unique_labels))

    def _get_unique_labels(self):
        '''
        Return a sorted list of unique labels
        '''
        return list(sorted(set(self.labels)))

'''packet_write_wait: Connection to 34.224.101.31 port 22: Broken pipe'''


def main():
    random.seed(39)
    img_proc = ImageProcessing(target_size=(150,150), max_qty=2)
    img_proc.pre_process_images()








if __name__ == '__main__':
    main()
