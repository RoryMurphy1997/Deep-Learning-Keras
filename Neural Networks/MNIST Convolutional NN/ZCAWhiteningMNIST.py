# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:41:05 2020

@author: RoryMurphy
"""
# ZCA Whitening: Highlights the structure and features in images to the learning algorithm.

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
datagen = ImageDataGenerator(samplewise_center=False, 
                             samplewise_std_normalization=False, 
                             featurewise_center=False,
                             featurewise_std_normalization=False,
                             zca_whitening=True)
datagen.fit(X_train)
# Retrieve one batch of images
for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=9):
    for i in range(0, 9):
        # 3X3 grid, plot number i+1
        pyplot.subplot(330+1+i)
        pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break
# Call fit_generator function



