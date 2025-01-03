import numpy as np


# 随机生成k个点作为质心，其中质心均在整个数据的边界之内
def rand_cent(data, k):
    n = data.shape[1]  # 获取数据的维度
    centroids = np.zeros((k, n))
    for j in range(n):
        range_j = np.max(data[:, j]) - np.min(data[:, j])
        centroids[:, j] = np.min(data[:, j]) + range_j * np.random.rand(k, )
    return centroids


# k-Means聚类算法,返回最终的k各质心和点的分配结果
def k_means(data, k, create_cent=rand_cent):
    m = data.shape[0]  # 获取样本数量
    # 构建一个簇分配结果矩阵，共两列，第一列为样本所属的簇类值，第二列为样本到簇质心的误差
    cluster_index = np.zeros(m, )
    # 1. 初始化k个质心
    centroids = create_cent(data, k)
    cluster_changed = True
    # 不断循环直到所有样本的分配不再产生变化
    while cluster_changed:
        cluster_changed = False
        # 2. 通过矩阵运算的方式计算所有样本到所有质心的距离
        data_matrix = np.expand_dims(data, 1)
        centroids_matrix = np.expand_dims(centroids, 0)
        dist = np.sqrt(np.sum((data_matrix - centroids_matrix) ** 2, axis=-1))
        # 3. 更新每一行样本所属的簇
        cluster_index_new = np.argmin(dist, axis=1)
        if (cluster_index_new != cluster_index).any():
            cluster_changed = True
            cluster_index = cluster_index_new
            # 4. 更新质心
            for center in range(k):
                data_per_cluster = data[np.argwhere(
                    cluster_index == center).squeeze()]  # 获取给定簇的所有点
                centroids[center, :] = np.mean(
                    data_per_cluster, axis=0)  # 求均值得到新的簇心
    return centroids, cluster_index
