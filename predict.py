import os
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('trained_model_93.33.h5')

os.chdir('extra_images/')

img = 'test_inter_2.jpg'

imag= Image.open(img)
imag = imag.resize((128, 128), Image.ANTIALIAS)
imag.save('new_'+img)

temp = []
temp.append(io.imread('new_'+img))
arr = np.array(temp)
arr = np.expand_dims(arr, axis=0)

def normalize(arr):
  arr = (arr.astype(np.float32)-127.5)/127.5
  arr = arr.reshape(arr.shape[0], 128,128, 3)
  return arr

arr = normalize(arr)

predictions = model.predict(arr)

predictions = list(predictions)

predictions = list(np.argmax(predictions, axis=1))

print(predictions[0])

plt.imshow(arr[0])

os.remove('new_'+img)

plt.show()