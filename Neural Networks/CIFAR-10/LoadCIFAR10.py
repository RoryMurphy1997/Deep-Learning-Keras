# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:48:15 2020

@author: RoryMurphy
"""


from keras.datasets import cifar10
import matplotlib.pyplot as pyplot
from PIL import Image

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
for i in range(0,9):
    pyplot.subplot(330+1+i)
    pyplot.imshow(Image.fromarray(X_train[i]))
pyplot.show()