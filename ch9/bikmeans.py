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


def rand_cent(data, k):
    """
    随机生成k个点作为质心，其中质心均在整个数据的边界之内
    """
    n = data.shape[1]  # 获取数据的维度
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = np.min(data[:, j])
        range_j = np.float(np.max(data[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids


def k_means(data, k, dist_meas=dist_eclud, create_cent=rand_cent):
    """
    k-Means聚类算法,返回最终的k各质心和点的分配结果
    """
    m = data.shape[0]  # 获取样本数量
    # 构建一个簇分配结果矩阵，共两列，第一列为样本所属的簇类值，第二列为样本到簇质心的误差
    cluster_assment = np.mat(np.zeros((m, 2)))
    # 1. 初始化k个质心
    centroids = create_cent(data, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            # 2. 找出最近的质心
            for j in range(k):
                dist_ji = dist_meas(centroids[j, :], data[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            # 3. 更新每一行样本所属的簇
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2
        print(centroids)  # 打印质心
        # 4. 更新质心
        for cent in range(k):
            pts_clust = data[np.nonzero(cluster_assment[:, 0].A == cent)[0]]  # 获取给定簇的所有点
            centroids[cent, :] = np.mean(pts_clust, axis=0)  # 沿矩阵列的方向求均值
    return centroids, cluster_assment


def bi_kmeans(data, k, dist_meas=dist_eclud):
    """
    二分k-Means聚类算法,返回最终的k各质心和点的分配结果
    """
    m = data.shape[0]
    cluster_assment = np.mat(np.zeros((m, 2)))
    # 创建初始簇质心
    centroid0 = np.mean(data, axis=0).tolist()[0]
    cent_list = [centroid0]
    # 计算每个点到质心的误差值
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(np.mat(centroid0), data[j, :]) ** 2
    while len(cent_list) < k:
        lowest_SSE = np.inf
        for i in range(len(cent_list)):
            # 获取当前簇的所有数据
            pts_in_curr_cluster = data[np.nonzero(cluster_assment[:, 0].A == i)[0], :]
            # 对该簇的数据进行K-Means聚类
            centroid_mat, split_clust_ass = k_means(pts_in_curr_cluster, 2, dist_meas)
            # 该簇聚类后的sse
            sse_split = sum(split_clust_ass[:, 1])
            # 获取剩余收据集的sse
            sse_not_split = sum(cluster_assment[np.nonzero(cluster_assment[:, 0].A != i)[0], 1])
            if (sse_split + sse_not_split) < lowest_SSE:
                best_cent_to_split = i
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass.copy()
                lowest_SSE = sse_split + sse_not_split
        # 将簇编号0,1更新为划分簇和新加入簇的编号
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split

        print("the best_cent_to_split is: ", best_cent_to_split)
        print("the len of best_clust_ass is: ", len(best_clust_ass))
        # 增加质心
        cent_list[best_cent_to_split] = best_new_cents[0, :]
        cent_list.append(best_new_cents[1, :])

        # 更新簇的分配结果
        cluster_assment[np.nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_clust_ass
    return cent_list, cluster_assment


if __name__ == '__main__':
    X = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
                  [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
                  [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
                  [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
                  [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
                  [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]],
                 dtype='float')
    k = 4  # 获取模拟数据和聚类数量

    my_centroids, cluster_assignment = bi_kmeans(X, k)  # my_centroids为簇质心
    centroids = np.array([i.A.tolist()[0] for i in my_centroids])  # 将matrix转换为ndarray类型
    # 获取聚类后的样本所属的簇值，将matrix转换为ndarray
    y_kmeans = cluster_assignment[:, 0].A[:, 0]
    # 未聚类前的数据分布
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='D', s=100, alpha=0.5)
    plt.title("用二分K-Means算法聚类的效果")
    plt.show()
    plt.savefig("./bikmeans.pdf")
    print('----------------------')
