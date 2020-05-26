# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:49:10 2020

@author: RoryMurphy
"""
from keras.models import model_from_json
import numpy
from PIL import Image
import matplotlib.pyplot as plt

# Load the image to predict
img = Image.open('handwritten_numbers/two_re.png').convert('L') # L mode = grayscale
img = img.resize((28,28))
im2arr = numpy.array(img)
im2arr = im2arr.reshape(1,1,28,28)
plt.imshow(img)
plt.show()

json_file = open('model_MNIST.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_MNIST.h5")
print("loaded model from disk.")

# Prediction
y_pred = loaded_model.predict(im2arr)
print(y_pred.argmax())