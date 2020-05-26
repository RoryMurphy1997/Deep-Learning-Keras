# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:10:16 2020

@author: RoryMurphy
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Define baseline model
# softmax: ensures output values are [0,1]
# categorical crossentropy: efficient Adam gradient descent optimization 
# algorithm with logarithmic loss function.
def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    return model

#Generate seed (allow random aspect to be reproduced)
seed = 7
numpy.random.seed(seed)

# Load Dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# Use one-hot encoding since there are three string output values
# 1,0,0 - Iris-setosa
# 0,1,0 - Iris-versicolor
# 0,0,1 - Iris-virginica
# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# Convert the integers to dummy variables
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=baseline_model,epochs=200, batch_size=5,
                            verbose=0)

# Evaluation
# Set number of folds = 10 and shuffle data before using (there is underlying
# order in this dataset)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results=cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Save model
estimator.fit(X, dummy_y)
model_json = estimator.model.to_json()
with open("model_iris.json", "w") as json_file:
    json_file.write(model_json)
    
estimator.model.save_weights("model_iris.h5")
print("Saved model to disk")