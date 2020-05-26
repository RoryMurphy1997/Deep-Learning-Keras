# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:31:21 2020

@author: RoryMurphy
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define baseline model
def create_baseline(optimizer='adam', init='normal'):
    # Draw the model: http://alexlenail.me/NN-SVG/index.html
    model = Sequential()
    model.add(Dense(60, input_dim=60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', 
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
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,epochs=100, 
                                          batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results=cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Save model
pipeline.fit(X, encoded_Y)
model_json = pipeline.model.to_json()
with open("model_sonar.json", "w") as json_file:
    json_file.write(model_json)

pipeline.model.save_weights("model_sonar.h5")
print("Saved model to disk")