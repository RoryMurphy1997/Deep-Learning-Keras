# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:23:06 2020

@author: RoryMurphy
"""


from keras.models import model_from_json
import numpy

# Load json and create model
json_file = open("model_reg.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model_reg.h5")
print("Loaded model from disk")

# Make prediction using loaded model
Xnew = [[6.3,2.5,5,1.9]]

# COnvert array to float 64 array
Xnew = numpy.array(Xnew, dtype=numpy.float64)

ynew = loaded_model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))