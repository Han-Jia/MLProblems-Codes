# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2022-04-18 12:49:35
"""

import numpy as np

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'


def load_data():
    # 以feature, label的形式返回数据集
    feature, label = datasets.load_iris(return_X_y=True)
    print(feature.shape)  # (150, 4)
    print(label.shape)  # (150,)
    feature_train, feature_test, label_train, label_test = \
        train_test_split(feature, label, test_size=0.2, random_state=0)
    return feature, label, feature_train, feature_test, label_train, label_test


def check_data():
    feature, label, feature_train, feature_test, label_train, label_test = load_data()

    # normalize feature_train
    feature_train_norm = (feature_train - np.min(feature_train, axis=0, keepdims=True)) / \
                         (np.max(feature_train, axis=0, keepdims=True) - np.min(feature_train, axis=0, keepdims=True))

    feature_train_norm_0 = feature_train_norm[label_train == 0]
    feature_train_norm_1 = feature_train_norm[label_train == 1]
    feature_train_norm_2 = feature_train_norm[label_train == 2]

    color_lists = ['#40DFEF', '#B9F8D3', '#E78EA9']
    line_color_lists = ['blue', 'green', 'red']

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(30, 7))
    plt.subplots_adjust(wspace=0.05)

    for i in range(0, 4):
        ax = axes[i]

        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)

        ax.set_xlabel('feature %d' % (i + 1), fontsize=40)
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([0, 1], fontsize=40)

        if i == 0:
            ax.set_ylabel('instance ratio', fontsize=40)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1], fontsize=40)

        weights_0 = np.ones_like(feature_train_norm_0[:, i]) / float(len(feature_train_norm_0[:, i]))
        weights_1 = np.ones_like(feature_train_norm_1[:, i]) / float(len(feature_train_norm_1[:, i]))
        weights_2 = np.ones_like(feature_train_norm_2[:, i]) / float(len(feature_train_norm_2[:, i]))

        ax.hist(feature_train_norm_0[:, i], range=(0, 1), bins=20, weights=weights_0, color=color_lists[0],
                alpha=0.5, edgecolor=line_color_lists[0], linewidth=2, label='class 0')
        ax.hist(feature_train_norm_1[:, i], range=(0, 1), bins=20, weights=weights_1, color=color_lists[1],
                alpha=0.5, edgecolor=line_color_lists[1], linewidth=2, label='class 1')
        ax.hist(feature_train_norm_2[:, i], range=(0, 1), bins=20, weights=weights_2, color=color_lists[2],
                alpha=0.5, edgecolor=line_color_lists[2], linewidth=2, label='class 2')

        if i == 3:
            ax.legend(loc='upper right', fontsize=40)

    plt.tight_layout()
    plt.show()


def test_gaussian_naive_bayes():
    feature, label, feature_train, feature_test, label_train, label_test = load_data()

    # 对训练集特征做min-max归一化
    feature_train_norm = (feature_train - np.min(feature_train, axis=0, keepdims=True)) / \
                         (np.max(feature_train, axis=0, keepdims=True) - np.min(feature_train, axis=0, keepdims=True))
    # 对测试集特征做min-max归一化
    feature_test_norm = (feature_test - np.min(feature_test, axis=0, keepdims=True)) / \
                        (np.max(feature_test, axis=0, keepdims=True) - np.min(feature_test, axis=0, keepdims=True))

    # 创建模型
    model = GaussianNB(priors=[1. / 3., 1. / 3., 1. / 3.])
    # 训练模型
    model.fit(feature_train_norm, label_train)
    # 预测
    label_pred = model.predict(feature_test_norm)
    # 计算准确率
    print('accuracy: %.2f' % sklearn.metrics.accuracy_score(label_test, label_pred))


if __name__ == "__main__":
    # check_data()
    test_gaussian_naive_bayes()
