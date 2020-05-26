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


def create_model():
    # Build the model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', 
                    activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    return model

# Load Dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")
#Split into input and output values
X = dataset[:,0:8]
Y = dataset[:,8]
#Generate seed (allow random aspect to be reproduced)
seed = 7
numpy.random.seed(seed)

model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10)
# Evaluate using 10 folds
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
