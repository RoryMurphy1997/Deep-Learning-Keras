# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:40:03 2020

@author: RoryMurphy
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
import numpy
from keras import backend as K
# Sets shape to (depth, input_depth, rows, columns)
K.set_image_dim_ordering('th')

def baseline_model():
    model = Sequential()
    # Hidden Layers:
    # Build 30 5X5 filters to apply to an input of a single dimension (gray 
    # scale) 28X28 image
    model.add(Conv2D(32, (3,3), padding='valid', kernel_constraint=maxnorm(3), 
                             input_shape=(3, 32, 32), activation='relu'))
    # Dropout 20% of the neurons on each iteration to prevent overfit
    model.add(Dropout(0.2))
    # Build 32 3X3 filters to apply
    model.add(Conv2D(32, (3,3), activation='relu', 
                     kernel_constraint=maxnorm(3)))
    # Pool 3X3 filters into 2X2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Classification Layers:
    # Flattens all the filters together back into one image
    model.add(Flatten())
    # Fully connected hidden layer with 512 neurons is used
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
     # Dropout 20% of the neurons on each iteration to prevent overfit
    model.add(Dropout(0.5))
    # Output layer identifies probability of image being each of the ten 
    # possible images.
    model.add(Dense(num_classes, activation='softmax'))
    # Compile Model
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    print(model.summary())
    return model

# Set seed
seed = 7
numpy.random.seed(seed)

# Load/Download the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize inputs
X_train = X_train/255
X_test = X_test/255

# One hot encoding outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

model = baseline_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=25, 
          batch_size=32)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Serialize model
model_json = model.to_json()
with open("model_cifar10.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_cifar10.h5")
print("Saved model")







