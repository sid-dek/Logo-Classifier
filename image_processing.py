import os
from PIL import Image
import numpy as np
import skimage.io as io

path = os.getcwd()
os.chdir(path + '/images')
path = os.getcwd()

width, height = 128, 128 

def resize():
    #width, height = 512, 512
 
    for each in os.listdir(path):
        print(each)
        raw_image = Image.open(each)
        compressed_image = raw_image.resize((width, height), Image.ANTIALIAS)
        compressed_image.save(each, quality = 75)

def save_data(data):
    np.save('data.npy', data)

def load_data():
    #os.chdir(os.pardir)
    data = np.load('data.npy')
    return data

def convert_gray():
    c = 0
    for each in os.listdir(path):
        img = Image.open(each).convert('1')
        img.save(each)
        c += 1

def image_to_array():
    count = 0
    temp = []
    labels = []
    # for each in os.listdir(path):
    for i in range(1, 301):
        count += 1
        temp.append(io.imread(str(i)+'.jpg'))
        if i <= 50:  #Barca
            labels.append(0)    
        elif i > 50 and i <= 100:
            labels.append(1)    #RMA
        elif i > 100 and i <= 150:
            labels.append(2)    #ManU
        elif i > 150 and i <= 200:
            labels.append(3)    #BVB
        elif i > 200 and i <= 250:
            labels.append(4)    #Inter
        else:
            labels.append(5)    #Chelsea
        print(i, labels[-1])
    arr = np.array(temp)
    print('Total images: ', count)
    print('Total labels: ', len(labels))
    return arr, labels
    
def create_labels(labels):
    labels_array = np.array(labels)
    np.save('labels', labels_array)

# resize()
data, labels = image_to_array()
data = (data.astype(np.float32) - 127.5)/ 127.5
data = data.reshape(data.shape[0], width, height, 3)
save_data(data)
create_labels(labels)