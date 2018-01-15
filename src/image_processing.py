import datetime as dt
import json
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from os import listdir
from os.path import isfile, join
from PIL import Image
import random
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import sys

IMAGE_DATA_PATH = '../data/bin-images/'
JSON_DATA_PATH = '../data/metadata/'

'''
The ImageProcessing class provides an object that analyses the raw data folders
to determine and store the following information:
    1) total number of image/json files
    2) labels for each image, list of unique labels, and list of missing labels

Functions:

    process_next_batch()
        reads a batch of images, converts image data to arrays, and resizes
        images to a common size.
    has_more()
        returns a boolean indicating if there are more images in the training
        data set to process.
    get_datagenerators_vX()
        returns training and validation image data generators for use by the
        model. The last term in the function name represents the version
        number as I expect to test multiple settings for the generators
        and with to retain settings that perform well.
'''
class ImageProcessing(object):

    def __init__(self, image_path, batch_size=1000, target_size=(150,150)):
        self.image_path = image_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.batch_index = 0
        self.img_list, self.json_list = self._get_file_name_lists()
        self.num_images = len(self.img_list)
        self.labels = self._extract_labels()
        # missing_labels and unique_labels are only used to determine number of
        # outputs on the fully connected layer
        self.missing_labels = self._get_missing_labels()
        self.unique_labels = self._get_unique_labels()
        # images are resized prior to split, but no other manipulation is done
        self.X_train, self.X_test, self.y_train, self.y_test = self._get_train_test_split()
        if self.batch_size > len(self.X_train):
            self.batch_size = len(self.X_train)
        pass

    def has_more(self):
        '''
        Returns True if there are more images to process.
        '''
        if self.batch_index < len(self.X_train):
            return True
        else:
            return False

    def process_next_batch(self):
        '''
        This function processes a batch of images by resizing to a common size
        and converting the image to an array. It returns a list of processed
        image arrays equal to the batch_size variable. This batch size is not
        the same as the batch size defined for the model. It is simply a limit
        on the number of images to be processed in this function call given
        there are more than 500k images. It is up to the calling funtion to
        decide how many images to process in one step.
        '''
        print('\nProcessing batch ', (self.batch_index//self.batch_size)+1, ' of ',
            self.num_images//self.batch_size, ' ... ...')
        start_time = dt.datetime.now()

        # list of numpy array's representing a batch of images
        image_batch = []
        for idx in range(self.batch_size):
            with open(IMAGE_DATA_PATH + self.X_train[self.batch_index + idx], 'r+b') as f:
                with Image.open(f) as image:
                    # width, height = image.size
                    # print('Width = ', width, ', Height = ', height)
                    X = img_to_array(image).astype(int)
                    # need to spec anti_aliasing=True ut current version of
                    # skimage.transform.resize has a bug
                    # with preserve_range=False I get numbers that are e-17
                    new_X = resize(X, (150, 150), mode='edge', preserve_range=True)
                    image_batch.append(new_X)
        self.batch_index += idx + 1

        stop_time = dt.datetime.now()
        print("Batch processing took ", (stop_time - start_time).total_seconds(), "s.\n")
        return image_batch

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
        for idx, filename in enumerate(self.json_list):
            with open(JSON_DATA_PATH + filename) as f:
                json_data = json.load(f)
                labels.append(json_data['EXPECTED_QUANTITY'])
        stop_time = dt.datetime.now()
        print("Extracting took ", (stop_time - start_time).total_seconds(), "s.\n")
        return labels

    def _get_file_name_lists(self):
        '''
        Return a list of image file names and JSON file names, randomly shuffled
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
        return img_file_list, json_file_list

    def _get_missing_labels(self):
        '''
        Return the integer quantities missing from the labels
        '''
        start, end = self.labels[0], self.labels[-1]
        return sorted(set(range(start, end + 1)).difference(self.labels))

    def _get_train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.img_list,
            self.labels, test_size=0.20, random_state=42)
        return X_train, X_test, y_train, y_test

    def _get_unique_labels(self):
        '''
        Return a sorted list of unique labels
        '''
        return list(sorted(set(self.labels)))




def main():
    # below is an example of how to use the ImageProcessing processing class.
    img_proc = ImageProcessing(image_path='../data/bin-images',
                               batch_size=10,
                               target_size=(150,150))
    while img_proc.has_more():
        images = img_proc.process_next_batch()
        print("Size of image batch = ", sys.getsizeof(images))


    # print(img_proc.unique_labels)
    # print(img_proc.missing_labels)
    # print(img_proc.X_train)
    # print(img_proc.y_train)








if __name__ == '__main__':
    main()
