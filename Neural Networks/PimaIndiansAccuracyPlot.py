# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:24:25 2020

@author: RoryMurphy
"""

import numpy
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

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

model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

#Find the plotting history
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, 
                    verbose=0)
print(history.history.keys())
# summarize the history for accuracy
fig1 = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize the history for loss
fig2=plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()