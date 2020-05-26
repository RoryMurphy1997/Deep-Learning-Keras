# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:24:25 2020

@author: RoryMurphy
"""

import numpy
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score




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
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

#Serialize model to JSON
model_yaml = model.to_yaml()
with open("model_pima.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
    
#Serialize weights to HDF5
model.save_weights("model_pima.h5")
print("Saved model to disk")
