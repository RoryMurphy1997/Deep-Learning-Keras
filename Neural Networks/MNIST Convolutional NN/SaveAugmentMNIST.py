# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:41:05 2020

@author: RoryMurphy
"""
# Random Flips: Trains the model to better handle flipped images.

import os
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Creates alterations to the images in the dataset to increase dataset's size.
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
datagen.fit(X_train)
os.makedirs('augmented_images')
# Retrieve one batch of images
for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=9, 
                                     save_to_dir='augmented_images', 
                                     save_prefix='aug_',
                                     save_format='png'):
    for i in range(0, 9):
        # 3X3 grid, plot number i+1
        pyplot.subplot(330+1+i)
        pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break
# Call fit_generator function



