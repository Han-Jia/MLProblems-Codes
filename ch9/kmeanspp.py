import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def dist_eclud(vec_a, vec_b):
    """
    计算两个向量的欧式距离
    """
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def initialize(dataset, k):
    ''' K-means++初始化质心
    :参数 data: 数据集
    :参数 k: 簇的个数
    :返回 centroids: 簇中心
    '''
    # 得到数据样本的维度
    n = np.shape(dataset)[1]
    # 初始化为一个(k, n)的全零矩阵
    centroids = np.mat(np.zeros((k, n)))
    # step1: 随机选择样本点之中的一个点
    centroids[0, :] = dataset[np.random.randint(dataset.shape[0]), :]
    # 迭代
    for c_id in range(k - 1):
        dist = []
        # 遍历所有点
        for i in range(dataset.shape[0]):
            point = dataset[i, :]
            d_max = 100
            # 扫描所有质心，选出该样本点与最近的类中心的距离
            for j in range(centroids.shape[0]):
                temp_dist = dist_eclud(point, centroids[j, :])
                d = min(d_max, temp_dist)
            dist.append(d)
        dist = np.array(dist)
        # 返回的是dist里面最大值的下标，对应的是上面循环中的i
        next_centroid = dataset[np.argmax(dist), :]
        centroids[c_id + 1, :] = next_centroid  # 选出了下一次的聚类中心，开始k+1循环
    return centroids


def k_means_plus(dataset, k, dist_meas=dist_eclud, create_cent=initialize):
    ''' k-means++聚类
    :参数 dataset: 输入的数据集
    :参数 k: 聚类的个数，可调
    :参数 dist_meas: 计算距离的方法，可调
    :参数 create_cent: 初始化质心的位置的方法，可调
    :返回: k个类质心的位置坐标
    :返回: 样本所处的类&到该类质心的距离
    '''
    # 获取数据集样本数
    m = np.shape(dataset)[0]
    # 初始化一个（m,2）全零矩阵，用来记录没一个样本所属类，距离类中心的距离
    cluster_assignment = np.mat(np.zeros((m, 2)))
    # 创建初始的k个质心向量
    centroids = create_cent(dataset, k)
    # 聚类结果是否发生变化的布尔类型
    cluster_changed = True
    # 终止条件：所有数据点聚类结果不发生变化
    while cluster_changed:
        # 聚类结果变化布尔类型置为False
        cluster_changed = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离为正无穷，最小距离对应的索引为-1
            min_dist = float('inf')
            min_index = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                dist_ji = dist_meas(centroids[j], dataset[i])
                # 如果距离小于当前最小距离
                if dist_ji < min_dist:
                    # 当前距离为最小距离，最小距离对应索引应为j(第j个类)
                    min_dist = dist_ji
                    min_index = j
            # 当前聚类结果中第i个样本的聚类结果发生变化：布尔值置为True，继续聚类算法
            if cluster_assignment[i, 0] != min_index:
                cluster_changed = True
            # 更新当前变化样本的聚类结果和平方误差
            cluster_assignment[i, :] = min_index, min_dist ** 2
        # 打印k-means聚类的质心
        # print(centroids)
        # 遍历每一个质心
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            pts_in_clust = dataset[np.nonzero(cluster_assignment[:, 0].A == cent)[0]]
            # 计算这些数据的均值(axis=0:求列均值)，作为该类质心向量
            centroids[cent, :] = np.mean(pts_in_clust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, cluster_assignment


if __name__ == '__main__':
    X = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
                  [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
                  [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
                  [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
                  [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
                  [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]],
                 dtype='float')
    k = 4  # 获取模拟数据和聚类数量
    my_centroids, clust_assing = k_means_plus(X, k)  # myCentroids为簇质心
    centroids = np.array([i.A.tolist()[0] for i in my_centroids])  # 将matrix转换为ndarray类型
    # 获取聚类后的样本所属的簇值，将matrix转换为ndarray
    y_kmeans = clust_assing[:, 0].A[:, 0]
    # 未聚类前的数据分布
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='red', marker='D', s=100, alpha=0.5)
    plt.title("用kmeans++算法聚类的效果")
    plt.show()
    plt.savefig("./kmeans++.pdf")
