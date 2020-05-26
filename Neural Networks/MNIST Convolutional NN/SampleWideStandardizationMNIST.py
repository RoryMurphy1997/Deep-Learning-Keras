# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:41:05 2020

@author: RoryMurphy
"""
# Sample-wise standardization: distribution of pixel values is changes such 
# that mean pixel value is 0 for each image and st dev is 1.
# Gives the effect of highlighting the digits within each image

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
datagen = ImageDataGenerator(samplewise_center=True, 
                             samplewise_std_normalization=True, 
                             featurewise_center=False,
                             featurewise_std_normalization=False)
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



