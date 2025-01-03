# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib
import matplotlib.pyplot as plt

COLOR_LIST = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
color_idx = 0


def pred_score(cls, X, y, neg_sample_num):
    global color_idx
    y_prob = cls.predict_proba(X)
    threshold_list = list(x / 100 for x in range(50, 100, 5))
    threshold_list.append(1 - 1 / (1 + 1200 / neg_sample_num))
    acc_list = []
    for i in threshold_list:
        threshold = i
        pred = (y_prob[:, 1] > threshold).astype(int)
        accuracy = (np.sum(pred == y)) / len(y)
        print("Threshold={}||Acc = {}".format(threshold, accuracy))
        acc_list.append(accuracy)

    max_acc = max(acc_list[0:-1])
    max_thr = threshold_list[acc_list.index(max_acc)]
    plt.plot(threshold_list[0:-1], acc_list[0:-1], c=COLOR_LIST[color_idx],
             label=u'负例数目={} 最高精度= {:.2f}'.format(neg_sample_num, max_acc), lw=3)
    plt.scatter(max_thr, max_acc, s=40, c=COLOR_LIST[color_idx])
    plt.scatter(threshold_list[-1], acc_list[-1], s=40, marker="v", c=COLOR_LIST[color_idx])
    color_idx += 1


# 构建平衡训练集
x_train_balance, y_train_balance = make_blobs(
    n_samples=[1200, 1200],  # 正负样本各自的数目
    centers=[[0.0, 0.0], [2.0, 2.0]],  # 正负样本各自类中心的位置
    cluster_std=[1.5, 1.5],  # 正负样本各自的方差
    random_state=1,  # 固定随机种子
    shuffle=False)  # 不打乱数据，便于后续筛选出不平衡数据集

# 构建平衡测试集
x_test_balance, y_test_balance = make_blobs(
    n_samples=[1200, 1200],
    centers=[[0.0, 0.0], [2.0, 2.0]],
    cluster_std=[1.5, 1.5],
    random_state=2,  # 使用与训练集不同的随机种子
    shuffle=False)

x_train_pos = x_train_balance[1200:]
y_train_pos = y_train_balance[1200:]
x_train_neg = x_train_balance[:1200]
y_train_neg = y_train_balance[:1200]

config = {
    'font.family': 'serif',
    'font.serif': ['kaiti'],
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
    'font.size': 18,
}
matplotlib.rc('axes', linewidth=2)
plt.rcParams.update(config)
plt.figure(figsize=(8, 6))

for i in range(100, 1100, 200):
    print("Neg Sample:{}".format(i))
    index_select = random.sample(list(range(1200)), i)
    x_train_neg_select = x_train_neg[index_select]
    y_train_neg_select = y_train_neg[index_select]
    x_train_imbalance = np.concatenate((x_train_pos, x_train_neg_select), 0)
    y_train_imbalance = np.concatenate((y_train_pos, y_train_neg_select), 0)

    clf_imbalance = LogisticRegression()
    clf_imbalance.fit(x_train_imbalance, y_train_imbalance)
    score = clf_imbalance.score(x_test_balance, y_test_balance)
    pred_score(clf_imbalance, x_test_balance, y_test_balance, i)
    y_porb = clf_imbalance.predict_proba(x_test_balance)
    fpr, tpr, thresholds = roc_curve(y_test_balance, y_porb[:, 1])

plt.xlabel(u"阈值")
plt.ylabel(u"精度")
# plt.title('不同阈值下的准确率')
plt.legend(loc="lower right")
plt.savefig("Threshold.pdf")
plt.show()
