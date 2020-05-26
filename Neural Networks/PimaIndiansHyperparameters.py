# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:25:37 2020

@author: RoryMurphy
"""
import numpy
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


#Generate seed (allow random aspect to be reproduced)
seed = 7
numpy.random.seed(seed)

def create_model(optimizer='rmsprop', init='glorot_uniform'):
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

model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10)

#Hyperparameters: kernel_init, optimizer, epochs, 
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = numpy.array([50,100,150])
batches = numpy.array([5,10,20])

param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches
                  , init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)

# Summarize Results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))