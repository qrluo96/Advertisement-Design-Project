import numpy as np
import os
from sklearn.cluster import KMeans

features = np.load("../feature.npy")

features = np.reshape(features, (features.shape[0], -1))

estimator = KMeans(n_clusters=100, max_iter=1000, n_jobs=-1)
estimator.fit(features)
label_pred = estimator.labels_
print(label_pred)
centroids = estimator.cluster_centers_
print(centroids)
inertia = estimator.inertia_
print(inertia)