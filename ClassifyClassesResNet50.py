from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os

model = ResNet50(weights='imagenet')

path = '../Icon1Copy'
folders = os.listdir(path)
for folder in folders:
    if folder[0] == '.':
        continue
    data = np.zeros((1, 224, 224, 3))
    files = os.listdir(path+'/'+folder)
    for file in files:
        if os.path.splitext(file)[-1] != ".png":
            continue
        img = image.load_img(path+'/'+folder+'/'+file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        data = np.vstack((data, x))

    data = data[1:]
    # print(data.shape)

    preds = model.predict(data)
    log = open(path+'/'+folder+"/class.txt", "w")
    for i in range(data.shape[0]):
        logInfo = 'Predicted: ' + str(i+1) + " " + str(decode_predictions(preds, top=5)[i]) + '\n'
        log.write(logInfo)
    log.close()