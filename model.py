import csv
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout, ELU, AveragePooling2D
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocess import image_preprocess

def keras_nn(X_train, y_train, batch_size=128):
    model = Sequential()
    
    model.add(AveragePooling2D((2, 2), border_mode='valid', input_shape=(16, 32, 1)))
    model.add(Convolution2D(1, 2, 2, subsample=(1, 1)))
    model.add(ELU())
    model.add(MaxPooling2D((2, 4), border_mode='valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))  #

    model.summary()
    
    checkpoint = ModelCheckpoint("model.h5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4, verbose=1, mode='min')


    model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
    model_json = model.to_json()
    with open("model.json", "w") as model_file:
        model_file.write(model_json)
    
    history = model.fit(X_train, y_train, batch_size=32, nb_epoch=150, verbose=1, callbacks=[checkpoint, early_stop], validation_split=0.2, shuffle=True)

if __name__ == '__main__':
    X_train = np.load("x.data.npy")
    y_train = np.load("y.data.npy")
    keras_nn(X_train, y_train)

