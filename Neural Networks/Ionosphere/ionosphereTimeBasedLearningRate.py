# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:58:29 2020

@author: RoryMurphy
"""

# Tips with learning rate schedules (drop or time):
# - increase the initial learning rate (Results in larger changes to weights 
# at the begining, alowing for fine tuning later)
# - Use larger momentum: Helps algorithm to make updates in right direction 
# when the learning rate is decreasing
# - Try different rate schedules: See what works best for the specific problem

import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD # Stochastic gradient descent
from sklearn.preprocessing import LabelEncoder

# Fix random seed
seed = 7
numpy.random.seed(seed)

# Load dataset
dataframe = pandas.read_csv("ionosphere.csv", header=None)
dataset = dataframe.values

X = dataset[:,0:34].astype(float)
Y = dataset[:, 34]

# encode class value as integer
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(34, input_dim=34, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))

epochs = 50
learning_rate = 0.1
decay_rate = learning_rate/epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, 
          nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28)