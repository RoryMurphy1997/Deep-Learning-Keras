# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:31:21 2020

@author: RoryMurphy
"""

# Notes on dropout: Tends to work better for larger NN.
# Doesnt have to use 20%, but its a good place to start (can adjust the 
# hyperparameter if needed).
# Use a large learning rate and momentum.

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.constraints import maxnorm
from keras.optimizers import SGD

# Define baseline model
def create_baseline(optimizer='adam', init='normal'):
    # Draw the model: http://alexlenail.me/NN-SVG/index.html
    model = Sequential()
    # Dropout layer: .2 = one in five inputs will be randomly excluded from
    # each update cycle
    # A weight constraint is set such that weights cannot be above 3 
    # (recommended for using this technique)
    model.add(Dense(60, input_dim=60, activation='relu', 
                    kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Changes to momentum and learning rate also recommended by paper (increase
    # learning rate by one order of magnitude)
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, 
                  metrics=['accuracy'])
    return model

#Generate seed (allow random aspect to be reproduced)
seed = 7
numpy.random.seed(seed)

# Load Dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Data standardization: Used to improve model performance
# Make value ranges consistent with one another (Generally use lower ranges as
# opposed to higher ones)

# To prevent the algorithm having knowledge of unseen data duyring evaluation,
# perform standardization only to training data portion of each fold of k-fold
# cross-validation (for each iteration, standardize training folds, but not 
# test fold. Use pipeline to do this.
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,epochs=350, 
                                          batch_size=17,verbose=0)))
pipeline=Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results=cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
