import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

features = np.load("../feature.npy")

features = np.reshape(features, (features.shape[0], -1))

estimator = MiniBatchKMeans(n_clusters=100, batch_size=10)
# max_iter=1000

print("start fitting")
estimator.fit(features)
print("finished")
label_pred = estimator.labels_
print(label_pred)
# centroids = estimator.cluster_centers_
# print(centroids)
# inertia = estimator.inertia_
# print(inertia)

indexlist = []
for i in range(100):
    index = [j for j in range(len(label_pred)) if label_pred[j] == 0]
    indexlist.append(index)

print(indexlist[0])