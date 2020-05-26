# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:41:05 2020

@author: RoryMurphy
"""


from keras.datasets import mnist
from matplotlib import pyplot

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
for i in range(0, 9):
    # 3X3 grid, plot number i+1
    pyplot.subplot(330+1+i)
    pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()