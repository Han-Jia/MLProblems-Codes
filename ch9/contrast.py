import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from gen_dataset import X_convex, X_nonconvex, X_noise

if __name__ == '__main__':
    # 读取数据集
    datasets = [X_convex, X_nonconvex, X_noise]
    # 准备绘图
    plt.figure(figsize=(21, 13))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )
    plot_num = 1

    for i_dataset, X in enumerate(datasets):
        # 调用函数实现所有聚类方法
        #kmeans = KMeans(n_clusters=2, random_state=0)
        #min = AgglomerativeClustering(n_clusters=2, linkage="single")
        #avg = AgglomerativeClustering(n_clusters=2, linkage="average")
        #max = AgglomerativeClustering(n_clusters=2, linkage="complete")
        #dbscan = DBSCAN(eps=0.5)
        
        # 准备好方法对应的名称
        clustering_algorithms = (
            ("KMeans", kmeans),
            ("Min_dist\nAgnes", min),
            ("Avg_dist\nAgnes", avg),
            ("Max_dist\nAgnes", max),
            ("DBSCAN", dbscan),
        )
        # 绘制聚类结果
        for name, algorithm in clustering_algorithms:
            res = algorithm.fit(X)
            y_pred = res.labels_
            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
            data_rgb = matplotlib.colors.ListedColormap(['#1E90FF', '#E3170D'])
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap=data_rgb)  # 绘制数据点
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1
    plt.savefig('./result_contrast.pdf')
