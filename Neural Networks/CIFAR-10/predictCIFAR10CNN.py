# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:22:19 2020

@author: RoryMurphy
"""

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img

# Load json and create model
json_file = open('model_cifar10.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model_cifar10.h5")
print("Loaded model")

# Make predictions
cifar_classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']
# Load the image to predict
img = load_img('images/birb2.jpg')
img = img.resize((32,32))
im2arr = img_to_array(img)
im2arr = im2arr.reshape((1,) + im2arr.shape)


# Prediction
y_pred = loaded_model.predict(im2arr)
print("The object appearing in the image is a:")
print(str([cifar_classes[y_pred.argmax()]]))