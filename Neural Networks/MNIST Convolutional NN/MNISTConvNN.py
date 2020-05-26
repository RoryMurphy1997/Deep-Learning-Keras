# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:54:32 2020

@author: RoryMurphy
"""

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
# Sets shape to (depth, input_depth, rows, columns)
K.set_image_dim_ordering('th')

def baseline_model():
    model = Sequential()
    # Hidden Layers:
    # Build 32 5X5 filters to apply to an input of a single dimension (gray 
    # scale) 28X28 image
    model.add(Conv2D(32, (5,5), padding='valid', 
                             input_shape=(1, 28, 28), activation='relu'))
    # Pool 5X5 filters into 2X2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout 20% of the neurons on each iteration to prevent overfit
    model.add(Dropout(0.2))
    # Classification Layers:
    # Flattens all the filters together back into one image
    model.add(Flatten())
    # Hidden layer with 128 neurons is used
    model.add(Dense(128, activation='relu'))
    # Output layer identifies probability of image being each of the ten 
    # possible images.
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model
# Set seed
seed = 7
numpy.random.seed(seed)

# Load/Download the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape to be of size [samples][channels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# One hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

model = baseline_model()

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, 
          batch_size=200, verbose=2)
scores= model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))