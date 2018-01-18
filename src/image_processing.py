from collections import Counter
import datetime as dt
import json
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
from PIL import Image
import random
from resizeimage import resizeimage
import simplejson
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

    process_next_training_batch()
        reads a batch of images, converts image data to arrays, and resizes
        images to a common size.
    has_more_training_data()
        returns a boolean indicating if there are more images in the training
        data set to process.
    get_datagenerators_vX()
        returns training and validation image data generators for use by the
        model. The last term in the function name represents the version
        number as I expect to test multiple settings for the generators
        and with to retain settings that perform well.
'''
class ImageProcessing(object):

    def __init__(self, batch_size=1000, target_size=(150,150), max_qty=None):
        # self.image_path = image_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.max_qty = max_qty
        self.train_index = 0
        self.test_index = 0
        # get an array of image and json file names, randomly shuffled with
        # outliers filtered out.
        self.image_files, self.json_files = self._get_file_name_arrays()
        self.num_images = len(self.image_files)
        # get an array of labels in the same shuffled order as the image and
        # json files with outliers removed
        self.labels = self._extract_labels()
        # missing_labels and unique_labels are only used to understand output
        # layer structure of the neural network
        self.unique_labels = self._get_unique_labels()
        self.missing_labels = self._get_missing_labels()
        # split the data into train and test before any processing
        # self.train_img, self.test_img, self.train_lbl, self.test_lbl = \
        #     train_test_split(self.image_files,
        #                      self.labels,
        #                      test_size=0.20,
        #                      random_state=42)
        pass

    def has_more_test_data(self):
        '''
        Returns True if there are more images to process.
        '''
        if self.test_index < len(self.test_img):
            return True
        else:
            return False

    def has_more_training_data(self):
        '''
        Returns True if there are more images to process.
        '''
        if self.train_index < len(self.train_img):
            return True
        else:
            return False

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
        self.train_img, self.test_img, self.train_lbl, self.test_lbl = \
            train_test_split(self.image_files,
                             self.labels,
                             test_size=0.20,
                             random_state=39)

        # create the processed training image array. Pixel values saved are
        # uint8 to save space. Normalization needs to be done in the model.
        print('\nPre-processing training images ... ...')
        start_time = dt.datetime.now()

        depth = 3
        arr = np.zeros((len(self.train_img), self.target_size[0], self.target_size[1], depth), dtype=np.uint8)
        for idx, img in enumerate(self.train_img):
            with open(IMAGE_DATA_PATH + img, 'r+b') as f:
                with Image.open(f) as image:
                    resized_image = resizeimage.resize_contain(image, self.target_size)
                    resized_image = resized_image.convert("RGB")
                    #resized_image.save(IMAGE_DATA_PATH + 'resized-' + self.X_train[self.batch_index], image.format)
                    X = img_to_array(resized_image).astype(np.uint8)
                    arr[idx] = X
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
        arr = np.zeros((len(self.test_img), self.target_size[0], self.target_size[1], depth), dtype=np.uint8)
        for idx, img in enumerate(self.test_img):
            with open(IMAGE_DATA_PATH + img, 'r+b') as f:
                with Image.open(f) as image:
                    resized_image = resizeimage.resize_contain(image, self.target_size)
                    resized_image = resized_image.convert("RGB")
                    #resized_image.save(IMAGE_DATA_PATH + 'resized-' + self.X_train[self.batch_index], image.format)
                    X = img_to_array(resized_image).astype(np.uint8)
                    arr[idx] = X
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

        print("Size of training labels numpy array = ", sys.getsizeof(self.train_lbl))
        np.save('../../dsi-capstone-data/training_labels.npy', self.train_lbl)
        print("Size of test labels numpy array = ", sys.getsizeof(self.test_lbl))
        np.save('../../dsi-capstone-data/training_labels.npy', self.test_lbl)

        stop_time = dt.datetime.now()
        print("Saving arrays took ", (stop_time - start_time).total_seconds(), "s.\n")

        pass

    def process_next_test_batch(self):
        '''
        This function processes a batch of images by resizing to a common size
        and converting the image to an array. It returns a list of processed
        image arrays equal to the batch_size variable. This batch size is not
        the same as the batch size defined for the model. It is simply a limit
        on the number of images to be processed in this function call given
        there are more than 500k images. It is up to the calling funtion to
        decide how many images to process in one step.
        '''
        print('\nProcessing test batch index', self.test_index, '(out of',
            len(self.test_img), 'total test images) ... ...')
        start_time = dt.datetime.now()

        # list of numpy array's representing a batch of images
        image_batch = []
        label_batch = []
        for idx in range(self.batch_size):
            if len(self.test_img) > self.test_index:
                with open(IMAGE_DATA_PATH + self.test_img[self.test_index], 'r+b') as f:
                    with Image.open(f) as image:
                        resized_image = resizeimage.resize_contain(image, self.target_size)
                        resized_image = resized_image.convert("RGB")
                        #resized_image.save(IMAGE_DATA_PATH + 'resized-' + self.X_train[self.batch_index], image.format)
                        X = img_to_array(resized_image).astype(int)
                        image_batch.append(X)
                label_batch.append(self.test_lbl[self.test_index])
                self.test_index += 1

        stop_time = dt.datetime.now()
        print("Batch processing took ", (stop_time - start_time).total_seconds(), "s.\n")
        return np.asarray(image_batch), np.asarray(label_batch)

    def process_next_training_batch(self):
        '''
        This function processes a batch of images by resizing to a common size
        and converting the image to an array. It returns an array of processed
        image arrays equal to the batch_size variable. This batch size is not
        the same as the batch size defined for the model. It is simply a limit
        on the number of images to be processed in this function call given
        there are more than 500k images. It is up to the calling funtion to
        decide how many images to process in one step.
        '''
        print('\nProcessing training batch index', self.train_index, '(out of',
            len(self.train_img), 'total training images) ... ...')
        start_time = dt.datetime.now()

        # list of numpy array's representing a batch of images
        image_batch = []
        label_batch = []
        for idx in range(self.batch_size):
            if len(self.train_img) > self.train_index:
                with open(IMAGE_DATA_PATH + self.train_img[self.train_index], 'r+b') as f:
                    with Image.open(f) as image:
                        resized_image = resizeimage.resize_contain(image, self.target_size)
                        resized_image = resized_image.convert("RGB")
                        #resized_image.save(IMAGE_DATA_PATH + 'resized-' + self.X_train[self.batch_index], image.format)
                        X = img_to_array(resized_image).astype(np.uint8)
                        image_batch.append(X)
                label_batch.append(self.train_lbl[self.train_index])
                self.train_index += 1

        stop_time = dt.datetime.now()
        print("Batch processing took ", (stop_time - start_time).total_seconds(), "s.\n")
        return np.asarray(image_batch), np.asarray(label_batch)

    def get_datagenerators_v1(self):
        '''
        Define the image manipulation steps to be randomly applied to each
        image. Multiple versions of this function will likely exist to test
        different strategies.
        Return a generator for both train and test data.
        '''
        train_generator = idg(featurewise_center=False, # default
            samplewise_center=False,                    # default
            featurewise_std_normalization=False,        # default
            samplewise_std_normalization=False,         # default
            zca_whitening=False,                        # default
            zca_epsilon=1e-6,                           # default
            rotation_range=0.,                          # default
            width_shift_range=0.,                       # default
            height_shift_range=0.,                      # default
            shear_range=0.,                             # default
            zoom_range=0.,                              # default
            channel_shift_range=0.,                     # default
            fill_mode='nearest',                        # default
            cval=0.,                                    # default
            horizontal_flip=False,                      # default
            vertical_flip=False,                        # default
            rescale=1./255,                             # rescale RGB vales
            preprocessing_function=None,                # default
            data_format='channels_last')                # default
        test_generator = idg(featurewise_center=False,  # default
            samplewise_center=False,                    # default
            featurewise_std_normalization=False,        # default
            samplewise_std_normalization=False,         # default
            zca_whitening=False,                        # default
            zca_epsilon=1e-6,                           # default
            rotation_range=0.,                          # default
            width_shift_range=0.,                       # default
            height_shift_range=0.,                      # default
            shear_range=0.,                             # default
            zoom_range=0.,                              # default
            channel_shift_range=0.,                     # default
            fill_mode='nearest',                        # default
            cval=0.,                                    # default
            horizontal_flip=False,                      # default
            vertical_flip=False,                        # default
            rescale=1./255,                             # rescale RGB vales
            preprocessing_function=None,                # default
            data_format='channels_last')                # default
        return train_generator, test_generator

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




def main():
    # below is an example of how to use the ImageProcessing processing class.
    # img_proc = ImageProcessing(batch_size=10,
    #                            target_size=(150,150),
    #                            qty_limit=None)
    # while img_proc.has_more_training_data():
    #     images, labels = img_proc.process_next_training_batch()
    #     print("Images array shape = ", images.shape)
    #     print("Size of image batch = ", sys.getsizeof(images))
    #     print(labels)

    random.seed(39)
    img_proc = ImageProcessing(batch_size=10,
                               target_size=(150,150),
                               max_qty=2)
    img_proc.pre_process_images()

    # print(img_proc.unique_labels)
    # print(img_proc.missing_labels)
    # print(img_proc.X_train)
    # print(img_proc.y_train)

    # with open('../data/img_proc.txt', 'w') as f:
    #     f.write("Labels:\n")
    #     simplejson.dump(img_proc.labels, f)
    #     f.write("\nUnique Labels:\n")
    #     simplejson.dump(img_proc.unique_labels, f)
    #     f.write("\nMissing Labels:\n")
    #     simplejson.dump(img_proc.missing_labels, f)
    #     f.write("\nLabel counts:\n")
    #     simplejson.dump(Counter(img_proc.labels), f)
    #
    # pickle.dump(img_proc.labels, open( "../data/labels.pkl", "wb" ))
    # pickle.dump(img_proc.image_files, open( "../data/image_list.pkl", "wb" ))
    # pickle.dump(img_proc.json_files, open( "../data/json_files.pkl", "wb" ))







if __name__ == '__main__':
    main()
