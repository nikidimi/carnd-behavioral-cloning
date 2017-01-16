import csv
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout, ELU, AveragePooling2D
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocess import image_preprocess

def keras_nn(X_train, y_train):
    """
    Constructs a neural network using keras and trains it.
    The best final networks is saved in model.json and model.h5

    Parameters
    ----------
    X_train : numpy array
        The images
    y_train : numpy array
        The angles
        
    """
    model = Sequential()
    
    # Further reduces the dimension of the image to 8x16
    model.add(AveragePooling2D((2, 2), border_mode='valid', input_shape=(16, 32, 1)))
    # Applies 2x2 convolution
    model.add(Convolution2D(1, 2, 2, subsample=(1, 1)))
    model.add(ELU())
    # Max Pooling to reduce the dimensions. 2X4 used because it matches the aspect ratio of the input
    model.add(MaxPooling2D((2, 4), border_mode='valid'))
    # Droput - We only have 10 connections at this point, but it still improves performance. However it should be kept low, e.g. 0.5 doesn't work
    model.add(Dropout(0.25))
    model.add(Flatten())
    # The final layer - outputs a float number (the steering angle)
    model.add(Dense(1))  #

    # Show a summary of the neural network
    model.summary()
    
    # Save the best model by validation mean squared error
    checkpoint = ModelCheckpoint("model.h5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
    
    # Stop training when there is no improvment. 
    # This is to speed up training, the accuracy is not affected, because the checkpoint will pick-up the best model anyway
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4, verbose=1, mode='min')

    # Compile the model with Adam optimizer and monitor mean squared error
    model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
    
    # Save the model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as model_file:
        model_file.write(model_json)
    
    # Start training.
    # nb_epoch should be a big number, there is early stopping callback anyway
    # Data is split by keras to training and validation
    history = model.fit(X_train, y_train, batch_size=32, nb_epoch=150, verbose=1, callbacks=[checkpoint, early_stop], validation_split=0.2, shuffle=True)

if __name__ == '__main__':
    # Reads the images and angles from a numpy binary file
    X_train = np.load("x.data.npy")
    y_train = np.load("y.data.npy")
    
    # Trains the network
    keras_nn(X_train, y_train)

