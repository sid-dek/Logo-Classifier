import os
import numpy as np
from PIL import Image
import skimage.io as io

path = os.getcwd()
os.chdir(path + '/dataset/')
path = os.getcwd()

width, height = 128, 128

def resize():
    for each in os.listdir(path):
        image = Image.open(each)
        image = image.resize((width, height), Image.ANTIALIAS)
        image.save(each, quality=90)

def image_to_array():
    temp = []
    labels = []
    for i in range(1, 301):
        temp.append(io.imread(str(i)+'.jpg'))
        if i<=50 :
            labels.append(0)  # Barcelona
        elif i>50 and i<=100:
            labels.append(1)  # RM
        elif i>100 and i<=150:
            labels.append(2)  # Man U
        elif i>150 and i<=200:
            labels.append(3)  # Dortmund
        elif i>200 and i<=250:
            labels.append(4)  # Inter
        else:
            labels.append(5)  # Chelsea

    arr = np.array(temp)
    labels = np.array(labels)
    return arr,labels

def save_data(arr, name):
    np.save(name, arr)

arr, labels = image_to_array()

save_data(arr, 'data.npy')
save_data(labels, 'labels.npy')

