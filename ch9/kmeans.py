import numpy as np
from sklearn.cluster import KMeans


# 直接调用sklearn实现kmeans算法
def kmeans(data, n):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
    cluster_index = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return centroids, cluster_index
