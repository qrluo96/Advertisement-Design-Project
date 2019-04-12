from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from sklearn.cluster import KMeans

model = VGG16(weights='imagenet', include_top=False)

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
        x = preprocess_input(x)
        data = np.vstack((data, x))

data = data[1:]

features = model.predict(data)
print(features.shape)
print(features)

# estimator = KMeans(n_clusters=100, max_iter=1000, n_jobs=-1)
# estimator.fit(features)
# label_pred = estimator.labels_
# print(label_pred)
# centroids = estimator.cluster_centers_
# print(centroids)
# inertia = estimator.inertia_
# print(inertia)

