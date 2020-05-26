# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:18:35 2020

@author: RoryMurphy
"""


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define baseline model
def create_baseline_deeper():
    # Draw the model: http://alexlenail.me/NN-SVG/index.html
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_baseline_wider():
    # Draw the model: http://alexlenail.me/NN-SVG/index.html
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#Generate seed (allow random aspect to be reproduced)
seed = 7
numpy.random.seed(seed)

# Load Dataset
dataframe = pandas.read_csv("BostonHousing.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:13]
Y = dataset[:,13]

estimator = KerasRegressor(build_fn=create_baseline_deeper,epochs=100, 
                           batch_size=5,verbose=0)

# Evaluation
# Set number of folds = 10 and shuffle data before using (there is underlying
# order in this dataset)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results=cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (abs(results.mean()), results.std()))

# Standardisation
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=create_baseline_deeper, 
                                         epochs=100, batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results=cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized Results: %.2f (%.2f)" % (abs(results.mean()), results.std()))

estimatorTwo = KerasRegressor(build_fn=create_baseline_wider,epochs=100, 
                           batch_size=5,verbose=0)

# Evaluation
# Set number of folds = 10 and shuffle data before using (there is underlying
# order in this dataset)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results=cross_val_score(estimatorTwo, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (abs(results.mean()), results.std()))

# Standardisation
estimatorsTwo = []
estimatorsTwo.append(('standardize', StandardScaler()))
estimatorsTwo.append(('mlp', KerasRegressor(build_fn=create_baseline_wider, 
                                         epochs=100, batch_size=5,verbose=0)))
pipeline=Pipeline(estimatorsTwo)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results=cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized Results: %.2f (%.2f)" % (abs(results.mean()), results.std()))

# Save deeper model
estimator.fit(X, Y)
model_json = estimator.model.to_json()
with open("model_reg.json", "w") as json_file:
    json_file.write(model_json)
estimator.model.save_weights("model_reg.h5")
print("Saved deeper model to disk")

#Save Wider model
estimatorTwo.fit(X, Y)
model_json = estimatorTwo.model.to_json()
with open("model_reg_two.json", "w") as json_file:
    json_file.write(model_json)
estimatorTwo.model.save_weights("model_reg_two.h5")
print("Saved wider model to disk")