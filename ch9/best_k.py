import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 遍历所有需要考虑的k值，分别计算对应聚类结果下的SSE和SC
for k in [2, 3, 4, 5, 6, 7, 8]:
    # 直接调用sklearn实现kmeans聚类，输入的参数为聚类个数k，并且固定随机状态为0
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    cluster_index = kmeans.labels_  # 数据所属的簇
    centroids = kmeans.cluster_centers_  # 所有簇心
    # 通过矩阵运算的方式计算均方误差和SSE
    data_matrix = np.expand_dims(data, 1)
    centroids_matrix = np.expand_dims(centroids, 0)
    dist = np.sqrt(np.sum((data_matrix - centroids_matrix) ** 2, axis=-1))
    sse = np.sum(np.min(dist, axis=1))
    # 直接调用sklearn计算轮廓系数SC, 输入的参数为数据，数据所属的簇，并且固定随机状态为1
    sc = silhouette_score(data, cluster_index, random_state=1)
