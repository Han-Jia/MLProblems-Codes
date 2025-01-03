import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import random


# 可视化样本分布和分类超平面
def plot(i, j):
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    xlim = (-5.0, 9.0)
    ylim = (-5.0, 9.0)
    xx = np.linspace(xlim[0], xlim[1], 300)
    yy = np.linspace(ylim[0], ylim[1], 300)
    YY, XX = np.meshgrid(yy, xx)
    # 涂色
    X_new = np.c_[XX.ravel(), YY.ravel()]
    if j == 'imbalance':
        y_predict = clf_imbalance.predict(X_new).reshape(XX.shape)
    else:
        y_predict = clf_balance.predict(X_new).reshape(XX.shape)
    custom_cmap = ListedColormap(['#B0E0E6', '#FFC0CB'])
    plt.pcolormesh(xx, yy, y_predict, cmap=custom_cmap)

    # xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # if j =='imbalance':
    #     Z = clf_imbalance.decision_function(xy).reshape(XX.shape)
    # else:
    #     Z = clf_balance.decision_function(xy).reshape(XX.shape)
    # ax.contour(XX, YY, Z, colors="k", levels=[0], alpha=0.5, linestyles=["-"])

    ax = plt.gca()
    if i == 'test':
        plt.scatter(x_test_balance[1200:, 0], x_test_balance[1200:, 1], marker='_', c='tab:blue')
        plt.scatter(x_test_balance[:1200, 0], x_test_balance[:1200, 1], marker='+', c='tab:orange')
    else:
        if j == 'imbalance':
            plt.scatter(x_train_imbalance[1200:, 0], x_train_imbalance[1200:, 1], marker='_', c='tab:blue')
            plt.scatter(x_train_imbalance[:1200, 0], x_train_imbalance[:1200, 1], marker='+', c='tab:orange')
        else:
            plt.scatter(x_train_balance[1200:, 0], x_train_balance[1200:, 1], marker='_', c='tab:blue')
            plt.scatter(x_train_balance[:1200, 0], x_train_balance[:1200, 1], marker='+', c='tab:orange')
    # plt.show()
    plt.savefig('ch17_{}_{}.pdf'.format(j, i))


if __name__ == '__main__':
    random.seed(1)
    # 构建平衡训练集
    x_train_balance, y_train_balance = make_blobs(
        n_samples=[1200, 1200],
        centers=[[0.0, 0.0], [2.0, 2.0]],
        cluster_std=[1.5, 1.5],
        random_state=1,
        shuffle=False)
    # 构建不平衡训练集
    x_train_pos = x_train_balance[:1200]
    y_train_pos = y_train_balance[:1200]
    x_train_neg = x_train_balance[1200:]
    y_train_neg = y_train_balance[1200:]
    index_select = random.sample(list(range(1200)), 100)
    x_train_neg_select = x_train_neg[index_select]
    y_train_neg_select = y_train_neg[index_select]
    x_train_imbalance = np.concatenate((x_train_pos, x_train_neg_select), 0)
    y_train_imbalance = np.concatenate((y_train_pos, y_train_neg_select), 0)
    # 构建平衡测试集
    x_test_balance, y_test_balance = make_blobs(
        n_samples=[1200, 1200],
        centers=[[0.0, 0.0], [2.0, 2.0]],
        cluster_std=[1.5, 1.5],
        random_state=2,
        shuffle=False)
    # 分别在平衡训练集和不平衡训练集上训练svm模型
    clf_balance = LinearSVC(max_iter=5000)
    clf_balance.fit(x_train_balance, y_train_balance)
    clf_imbalance = LinearSVC(max_iter=5000)
    clf_imbalance.fit(x_train_imbalance, y_train_imbalance)
    for i in ['train', 'test']:
        for j in ['balance', 'imbalance']:
            plot(i, j)
