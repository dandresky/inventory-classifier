'''
This model is developed for development purposes only. I use it to test my
image processing and explore fully connected output configurations using a
smaller model architecture that can train relatively faster than the Deep
learning model I intend to use.

Model A uses a single output layer with a RELU activation function.
'''
import datetime as dt
import keras
from keras import losses
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator as idg
import numpy as np
from sklearn.metrics import mean_squared_error

def get_datagenerators_v1():
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

def get_model(filter_size=32, input_shape=(150,150,3)):
    model = Sequential()
    model.add(Conv2D(filter_size, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filter_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(2*filter_size, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(2*filter_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('relu'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    # Let's train the model using RMSprop
    model.compile(loss=losses.mean_squared_error,
                  optimizer=opt,
                  metrics=[metrics.mae])
    return model


def main():
    # get pre-processed image and label data
    print('\nLoading numpy arrays ... ...')
    start_time = dt.datetime.now()
    X_train = np.load('../../dsi-capstone-data/processed_training_images.npy')
    X_test = np.load('../../dsi-capstone-data/processed_test_images.npy')
    y_train = np.load('../../dsi-capstone-data/training_labels.npy')
    y_test = np.load('../../dsi-capstone-data/test_labels.npy')
    stop_time = dt.datetime.now()
    print("Loading arrays took ", (stop_time - start_time).total_seconds(), "s.\n")

    # create model
    model = get_model(filter_size=32, input_shape=(150,150,3))

    # data generators are instructions to Keras for further processing of the
    # image data (in batches) before training on the image.
    train_datagen, test_datagen = get_datagenerators_v1()
    train_datagen.fit(X_train)
    test_datagen.fit(X_test)

    # Fit the model on the batches generated by datagen.flow().
    # I am processing images in batches manually, rather than all at once,
    # therefore, epochs will default to 1
    batch_size = 10
    model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                        verbose=True,
                        epochs=4,
                        steps_per_epoch=(len(X_train)/batch_size),
                        workers=4)

    # Score trained model.
    score = model.evaluate(X_test, y_test,
                           batch_size=batch_size,
                           verbose=True,
                           sample_weight=None)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # make predictions and compare to actual
    pred = model.predict(X_test,
                         batch_size=batch_size,
                         verbose=True)
    print("Predictions:\n", pred)
    print("Actuals:\n", y_test)
    mse = mean_squared_error(pred, y_test)
    print("The MSE of the predicted quantities is", mse)
    pass












if __name__ == '__main__':
    main()
