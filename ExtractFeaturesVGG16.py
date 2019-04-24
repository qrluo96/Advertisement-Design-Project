from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from sklearn.cluster import KMeans

model = VGG16(weights='imagenet', include_top=False)

data = np.load("../data.npy")

features = model.predict(data)
print(features.shape)
print(features)

np.save("../feature.npy", features)

# estimator = KMeans(n_clusters=100, max_iter=1000, n_jobs=-1)
# estimator.fit(features)
# label_pred = estimator.labels_
# print(label_pred)
# centroids = estimator.cluster_centers_
# print(centroids)
# inertia = estimator.inertia_
# print(inertia)

