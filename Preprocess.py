from keras.preprocessing import image
import numpy as np
import os

path = '../Icon1Copy'
folders = os.listdir(path)
data = np.zeros((1, 224, 224, 3))
for folder in folders:
    if folder[0] == '.':
        continue
    files = os.listdir(path+'/'+folder)
    for file in files:
        if os.path.splitext(file)[-1] != ".png":
            continue
        img = image.load_img(path+'/'+folder+'/'+file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        data = np.vstack((data, x))

data = data[1:]

dataFile = open("../data.npy", "w")
np.save(dataFile, data)

