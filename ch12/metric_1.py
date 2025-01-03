import numpy as np
from sklearn.datasets import make_blobs
from sklearn import svm
import random

# 构建平衡训练集
x_train_balance, y_train_balance = make_blobs(
    n_samples=[1200, 1200],
    centers=[[0.0, 0.0], [2.0, 2.0]],
    cluster_std=[0.5, 0.5],
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
    cluster_std=[0.5, 0.5],
    random_state=2,
    shuffle=False)
# 分别在平衡训练集和不平衡训练集上训练svm模型
clf_balance = svm.SVC(kernel="linear", C=1.0)
clf_balance.fit(x_train_balance, y_train_balance)
clf_imbalance = svm.SVC(kernel="linear", C=1.0)
clf_imbalance.fit(x_train_imbalance, y_train_imbalance)
