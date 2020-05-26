# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:25:37 2020

@author: RoryMurphy
"""
import numpy
from sklearn.model_selection import train_test_split
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

# Define model: Sequential model (add one layer at a time), fully-connected
# structure with three layers

# Generally, determining the number of layers is to keep adding layers until
# the test error does not improve with the addition of the previous layer

# Dense class used to define a fully connected layer
# P1: no of layers, init = weight initialization method, 
# activation = activation function, determines how to transfer information
# to the next layer
# ReLu = Rectified linear units
# Sigmoid - used on output layer so that it outputs a binary value

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
# loss - loss function, binary crossentropy (uses logarithmic loss for a binary
# classification problem)
# adam: efficient gradient descent method
# metrics: any additional metrics to ouput. Since this is binary model, use
# accuracy
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

# Fit model: apply training data to it
# epochs: number of iterations used in training process
# batch size: number of iterations which occur before weights are updated
# These hyperparameters can be determines through trial and error
# Validation split determines the size of the test/validation set from the 
# given data
# Automatic Verification
# model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)

# Manual Verification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33,
                                                    random_state=seed)
model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=150,
          batch_size=10)

#Model Evaluation (normally split into training and test data)
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))