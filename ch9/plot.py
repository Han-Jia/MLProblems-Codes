import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# 绘制聚类结果，函数需要的输入分别为数据, 簇心，数据所属的簇，输出文件名
def plot(data, centroids, cluster_index, outname):
    # 分别确定横纵坐标的最小值和最大值
    x_min = np.min(data[:, 0]) - 0.02
    x_max = np.max(data[:, 0]) + 0.02
    y_min = np.min(data[:, 1]) - 0.02
    y_max = np.max(data[:, 1]) + 0.02
    # 控制遍历的步长，该值也可以理解为控制分辨率
    step = 0.001
    # 使用meshgrid函数遍历背景区域中的所有点，从而给背景着色
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    all_point = np.c_[xx.ravel(), yy.ravel()]
    # 通过矩阵运算的方式计算所有点到所有簇心的距离
    all_point_matrix = np.expand_dims(all_point, 1)
    centroids_matrix = np.expand_dims(centroids, 0)
    dist = np.sqrt(np.sum((all_point_matrix - centroids_matrix) ** 2, axis=-1))
    # 分别得到所有点的最近簇心
    color_index = np.argmin(dist, axis=1)
    z = color_index.reshape(xx.shape)
    # 分别控制背景色和数据点颜色
    bg_rgb = matplotlib.colors.ListedColormap(['#F5DEB3', '#BDFCC9', '#B0E0E6', '#FFC0CB'])
    data_rgb = matplotlib.colors.ListedColormap(['#FFD700', '#228B22', '#1E90FF', '#E3170D'])
    # 绘制背景
    plt.pcolormesh(xx, yy, z, cmap=bg_rgb)
    plt.scatter(data[:, 0], data[:, 1], c=cluster_index, s=50, cmap=data_rgb)  # 绘制数据点
    plt.scatter(centroids[:, 0], centroids[:, 1], c='#000000', marker='D', s=100)  # 绘制簇心
    plt.savefig(outname)
