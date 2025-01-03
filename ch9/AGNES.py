import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agnes(data, n):
    ag = AgglomerativeClustering(n_clusters=n).fit(data)
    cluster_index = ag.labels_
    centroids = []
    # 计算所有同簇样本的均值作为簇心
    for i in range(n):
        centroid = np.mean(data[np.argwhere(cluster_index == i).squeeze()], axis=0)
        centroids.append(np.expand_dims(centroid, axis=0))
    centroids = np.concatenate(centroids, axis=0)
    return centroids, cluster_index
