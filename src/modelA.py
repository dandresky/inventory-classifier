'''
This model is developed for development purposes only. I use it to test my
image processing and explore fully connected output configurations using a
smaller model architecture that can train relatively faster than the Deep
learning model I intend to use.

Model A uses a single output layer with a RELU activation function.
'''
import keras
from keras import losses
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from image_processing import ImageProcessing


def get_model(image_arr):
    model = Sequential()
    model.add(Conv2D(image_arr.shape[1], (3, 3), padding='same',
                     input_shape=image_arr.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(image_arr.shape[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(2*image_arr.shape[1], (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(2*image_arr.shape[1], (3, 3)))
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
    img_proc = ImageProcessing(batch_size=10,
                               target_size=(150,150))
    # process batches of images
    while img_proc.has_more():
        # get the next batch of processed images
        images = img_proc.process_next_batch()
        model = get_model(images)

    pass



if __name__ == '__main__':
    main()
