# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:23:06 2020

@author: RoryMurphy
"""


from keras.models import model_from_yaml
from numpy import array

# Load json and create model
yaml_file = open("model_pima.yaml", "r")
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)

# Load weights into new model
loaded_model.load_weights("model_pima.h5")
print("Loaded model from disk")

# Make prediction using loaded model
Xnew = array([[6,148,72,35,0,33.6,0.627,50]])

ynew = loaded_model.predict_classes(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))