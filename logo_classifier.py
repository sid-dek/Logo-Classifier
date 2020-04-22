# -*- coding: utf-8 -*-
"""Logo_Classifier.ipynb
"""

from google.colab import drive
drive.mount('/content/drive/')

import os
import numpy as np
import matplotlib.pyplot as plt

os.getcwd()

os.chdir('/content/drive/My Drive/Projects/Logo Classifier/')
os.getcwd()

os.listdir()

data = np.load('data.npy')
labels = np.load('labels.npy')

print(data.shape)
print(labels.shape)

np.amax(data)

def normalize(arr):
  arr = (arr.astype(np.float32)-127.5)/127.5
  arr = arr.reshape(arr.shape[0], 128,128, 3)
  return arr

data = normalize(data)

print(np.amax(data))
print(np.amin(data))
print(data.shape)

import os 
import math 
import random 
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from keras import optimizers 
from keras import applications 
from keras.layers import Input 
from keras import initializers 
import matplotlib.pyplot as plt 
from keras.optimizers import Adam 
from keras.models import load_model 
from keras.models import Model, Sequential 
from keras import regularizers, preprocessing 
from keras.layers.advanced_activations import LeakyReLU 
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D 
from keras.layers.core import Reshape, Dense, Dropout, Flatten, Activation

def split(arr):
  test_sample = [i for i in range(0,300,5)]
  train_sample = []
  
  for i in range(0, 300):
    if i not in test_sample:
      train_sample.append(i)
  
  temp_list = []
  temp = []
  
  for i in test_sample:
    temp_list.append(data[i])
    temp.append(labels[i])
  test_x = np.array(temp_list)
  test_y = np.array(temp)
  
  temp_list = []
  temp = []
  
  for i in train_sample:
    temp_list.append(data[i])
    temp.append(labels[i])
  train_x = np.array(temp_list)
  train_y = np.array(temp)
    
  return test_x , test_y, train_x, train_y
  
  
test_x , test_y, train_x, train_y = split(data)
print(test_x.shape)
print(test_y.shape)
print(train_x.shape)
print(train_y.shape)

def build_model():
  base_model = applications.vgg16.VGG16(include_top = False, weights='imagenet', input_shape = (128,128,3))
  for layer in base_model.layers[:-4]:
    layer.trainable = False
  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(256, activation = 'relu'))
  model.add(Dropout(0.75))
  model.add(Dense(6, activation = 'softmax'))
  model.compile(loss='sparse_categorical_crossentropy', optimizer = optimizers.Adam(lr=1e-4), metrics = ['accuracy'])
  return model
  
model = build_model()
model.summary()

train_x , train_y = shuffle(train_x , train_y)
test_x , test_y = shuffle(test_x , test_y)

history = model.fit(train_x , train_y, batch_size = 1, validation_split = 0.2, epochs = 25)

scores = model.evaluate(test_x , test_y)
print(model.metrics_names[1] , scores[1]*100)

plt.plot(history.history['acc']) 
plt.plot(history.history['val_acc']) 
plt.title('Model accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show() # Plot training & validation loss values 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

model.save('trained_model_93.33.h5')

