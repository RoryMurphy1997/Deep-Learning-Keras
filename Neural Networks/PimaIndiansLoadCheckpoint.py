# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:24:25 2020

@author: RoryMurphy
"""

import numpy
from keras.layers import Dense
from keras.models import Sequential


# Load Dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")
#Split into input and output values
X = dataset[:,0:8]
Y = dataset[:,8]
#Generate seed (allow random aspect to be reproduced)
seed = 7
numpy.random.seed(seed)

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', 
                    activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Load Checkpoint
model.load_weights("weights.best.hdf5")
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

print("Created a model using the loaded weights")