from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os

model = ResNet50(weights='imagenet')

path = '../Icon1Copy'
folders = os.listdir(path)
for folder in folders:
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


# # Read images
# path = '../Icon1Copy/SEO'
# files = os.listdir(path)
# data = np.zeros((1, 224, 224, 3))
# for file in files:
#     if os.path.splitext(file)[-1] != ".png":
#         continue
#     img = image.load_img(path+"/"+file, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     data = np.vstack((data, x))

# data = data[1:]
# print(data.shape)

# preds = model.predict(data)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# log = open(path+"/class.txt", "w")
# for i in range(data.shape[0]):
#     logInfo = 'Predicted: ' + str(i+1) + " " + str(decode_predictions(preds, top=5)[i]) + '\n'
#     log.write(logInfo)
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]