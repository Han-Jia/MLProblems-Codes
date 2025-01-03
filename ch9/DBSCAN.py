import numpy as np
from sklearn.cluster import DBSCAN


def dbscan(data, s):
    db = DBSCAN(eps=s).fit(data)
    cluster_index = db.labels_
    n_clusters = len(set(list(cluster_index))) - 1  # 被标记为-1的样本是模型认为的噪声样本，并不是一个簇
    centroids = []
    # 计算所有同簇样本的均值作为簇心，对于标记为-1的样本，不做处理
    for i in range(n_clusters):
        if data[np.argwhere(cluster_index == i).squeeze()].ndim == 1:
            centroid = data[np.argwhere(cluster_index == i).squeeze()]
        else:
            centroid = np.mean(data[np.argwhere(cluster_index == i).squeeze()], axis=0)
        centroids.append(np.expand_dims(centroid, axis=0))
    centroids = np.concatenate(centroids, axis=0)
    return centroids, cluster_index
