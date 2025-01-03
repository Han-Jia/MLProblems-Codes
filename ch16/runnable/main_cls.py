# @author : Administrator 
# @date : 2022/4/20

from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os

import matplotlib
from matplotlib import pyplot as plt

config = {
    'font.family': 'serif',
    'font.serif': ['kaiti'],
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
    'font.size': 14,
}
plt.rcParams.update(config)

matplotlib.rc('axes', linewidth=2)

bg_rgb = matplotlib.colors.ListedColormap([
    # '#F5DEB3','#BDFCC9',
    '#B0E0E6', '#FFC0CB'])
data_rgb = matplotlib.colors.ListedColormap([
    # '#FFD700','#228B22',
    '#1E90FF', '#E3170D'])

# 生成数据集
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                           random_state=0)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

margin = 0.3
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

K = 5
metrics = ['euclidean', 'seuclidean', 'mahalanobis']
variance = np.var(X_train, axis=0)
covariace = np.cov(X_train.T)

os.makedirs('figures', exist_ok=True)

metric_params = [None, {'V': variance}, {'V': covariace}]
for m, p in zip(metrics, metric_params):
    clf = KNeighborsClassifier(K, metric=m, metric_params=p)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=bg_rgb, alpha=1.)
    markers = ['+', '_']
    for i in range(2):
        idx = y_train == i
        plt.scatter(X_train[idx][:, 0], X_train[idx][:, 1], c=data_rgb.colors[i], s=30, marker=markers[i])
        idx = y_test == i
        plt.scatter(X_test[idx][:, 0], X_test[idx][:, 1], c=data_rgb.colors[i], s=30, marker=markers[i], alpha=0.75,
                    linestyle='dashed')
    # plt.ylabel("Feature #1")
    # plt.xlabel("Feature #0")
    plt.xticks([])
    plt.yticks([])
    plt.title(f"k-Nearest Neighbor ({m.title()})", font='sans serif')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.text(
        xx.max() - 0.1,
        yy.min() + 0.1,
        ("测试精度 : %.2f" % score).lstrip("0"),
        size=14,
        horizontalalignment="right",
    )
    plt.savefig(os.path.join('figures', f'ch16_cls_metric={m}.pdf'), transparent=True, bbox_inches='tight')
    plt.close(fig)
