# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:23:06 2020

@author: RoryMurphy
"""


from keras.models import model_from_json
from numpy import array

# Load json and create model
json_file = open("model_iris.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model_iris.h5")
print("Loaded model from disk")

# Make prediction using loaded model
Xnew = array([[6.3,2.5,5,1.9]])
nameArray = array(["Iris Setosa", "Iris Versicolor", "Iris Virginica"])

ynew = loaded_model.predict_classes(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], nameArray[ynew[0]]))