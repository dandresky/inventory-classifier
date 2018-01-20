import numpy as np


class ModelA(object):

    def __init__(self, X_train, y_train, X_test, y_test, batch_size=32):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        pass





def main():
    # get pre-processed image and label data
    print('\nLoading numpy arrays ... ...')
    X_train = np.load('../../dsi-capstone-data/processed_training_images.npy')
    X_test = np.load('../../dsi-capstone-data/processed_test_images.npy')
    y_train = np.load('../../dsi-capstone-data/training_labels.npy')
    y_test = np.load('../../dsi-capstone-data/test_labels.npy')












if __name__ == '__main__':
    main()
