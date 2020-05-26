# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:14:22 2020

@author: RoryMurphy
"""


from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load/Download the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Plot four images in a 2X2 grid
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show