import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# 读取数据
X = np.array([[0.46], [0.376], [0.264], [0.318], [0.215], [0.237], [0.149], [0.211],
              [0.091], [0.267], [0.057], [0.099], [0.161], [0.198], [0.170], [0.042], [0.103]])
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# 直接调用sklearn实现kmeans算法
def kmeans(data, n, init):
    kmeans = KMeans(n_clusters=n, random_state=0, init=init).fit(data)
    cluster_index = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return centroids, cluster_index


# 直接调用sklearn实现agnes算法
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


# 分别实现三个聚类
centroids_1, cluster_index_1 = kmeans(X, 2, np.array([[2.5, ], [1.5, ]]))
centroids_2, cluster_index_2 = kmeans(X, 2, np.array([[4.0, ], [0.0, ]]))
centroids_3, cluster_index_3 = agnes(X, 2)
