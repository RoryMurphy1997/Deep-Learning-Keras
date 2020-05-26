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
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD # Stochastic gradient descent
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler

# Learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop=0.5
    epochs_drop=10
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate


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

sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Learning schedule callback
lrate = LearningRateScheduler(step_decay)
callback_list = [lrate]
model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=18, 
          callbacks=callback_list)