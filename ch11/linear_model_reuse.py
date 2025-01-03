#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
-----------------------------------------------------------
------------------------ 第十一章  -------------------------
-----------------------------------------------------------
'''

import numpy as np
from sklearn.datasets import fetch_lfw_people  # 载入sklearn数据集相关函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 准确率计算函数

## get data and pre-processing
lfw_people = fetch_lfw_people(min_faces_per_person=60)  # 获取数据集，每类样本量不低于60
X, y = lfw_people.data, lfw_people.target  # 得到样例、标记矩阵
n_samples, n_features = X.shape
n_classes = lfw_people.target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print("image shape {} by {}".format(lfw_people.images.shape[1], lfw_people.images.shape[2]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2022  # 25%数据作为测试集
)
# 获得标记的one-hot编码
y_train_coding = - np.ones((y_train.size, n_classes))
y_train_coding[np.arange(y_train.size), y_train] = 1
y_test_coding = - np.ones((y_test.size, n_classes))
y_test_coding[np.arange(y_test.size), y_test] = 1

'''
-----------------------------------------------------------
------------------------   11.1   -------------------------
-----------------------------------------------------------
'''
## 实现岭回归分类器的闭式解求解函数
# def ridgeReg(X, Y, lmbd):
# n_samples, n_features = X.shape
## 构建中心化矩阵H
# H = np.identity(n_samples) - np.ones((n_samples, n_samples)) / n_samples
# HX = H @ X
## 基于闭式解公式求解W和b
# W = np.linalg.pinv(HX.T @ HX / n_samples  + 2 * lmbd * np.identity(n_features)) @ (HX.T @ Y) / n_samples
# b = np.mean(Y - (X@W), 0)
# return W, b

# lmbd = 1
## 使用闭式解函数，基于训练数据求得参数W和b
# W, b = ridgeReg(X_train, y_train_coding, lmbd)
## 针对测试集进行预测，使用置信度最大的索引作为预测类别
# y_pred = np.argmax(X_test@W + b, 1)
# acc_rls = accuracy_score(y_test, y_pred)                # 计算准确率
# print('RLS with closed-form solver gets accuracy {}'.format(acc_rls))

# from sklearn.linear_model import RidgeClassifier        # 岭回归用于分类
# clf = RidgeClassifier(random_state=0)                   # 以默认参数构建岭回归分类器
# clf.fit(X_train, y_train)                               # 基于训练数据优化模型参数
# y_pred = clf.predict(X_test)                            # 基于训练好的模型对测试集进行预测
# rls_acc = accuracy_score(y_test, y_pred)                # 利用测试集预测标记与真实标记计算准确率
# print('RLS gets accuracy {}'.format(rls_acc))

# from sklearn.svm import LinearSVC                       # 线性支持向量机分类模型
# clf = LinearSVC(random_state=0)                         # 以默认参数构建线性SVM分类器
# clf.fit(X_train, y_train)                               # 基于训练数据优化模型参数
# y_pred = clf.predict(X_test)                            # 基于训练好的模型对测试集进行预测
# svm_acc = accuracy_score(y_test, y_pred)                # 利用测试集预测标记与真实标记计算准确率
# print('SVM gets accuracy {}'.format(svm_acc))


from sklearn.preprocessing import MinMaxScaler  # 归一化函数

scaler = MinMaxScaler()  # 归一化数据到区间[0, 1]
X_train = scaler.fit_transform(X_train)  # 在训练集计算最大最小值
X_test = scaler.transform(X_test)  # 用训练集统计量处理测试集数据

# clf = LinearSVC(random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# svm_acc = accuracy_score(y_test, y_pred)
# print('After MinMaxScaler, SVM gets accuracy {}'.format(svm_acc))

# clf = RidgeClassifier(random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# rls_acc = accuracy_score(y_test, y_pred)
# print('After MinMaxScaler, RLS gets accur acy {}'.format(rls_acc))

from sklearn.decomposition import PCA

n_components = 160
pca = PCA(n_components=n_components).fit(X_train)  # PCA降维至160，基于训练数据获得投影矩阵
X_train = pca.transform(X_train)  # 对训练数据进行投影
X_test = pca.transform(X_test)  # 对测试数据进行投影
n_features = n_components

# clf = LinearSVC(random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# svm_acc = accuracy_score(y_test, y_pred)
# print('After PCA, SVM gets accuracy {}'.format(svm_acc))

# clf = RidgeClassifier(random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# rls_acc = accuracy_score(y_test, y_pred)
# print('After PCA, RLS gets accuracy {}'.format(rls_acc))

'''
-----------------------------------------------------------
------------------------   11.2   -------------------------
-----------------------------------------------------------
'''
# from scipy.optimize import minimize                     # 引入优化函数
## 参数初始化
# W0 = np.random.rand(n_features, n_classes)              # 随机初始化W
# b0 = np.zeros((n_classes))                              # 用全0向量初始化b
## 根据函数要求，求向量，因此将矩阵转为向量、拼接后作为优化变量
# m0 = np.concatenate([W0.ravel(), b0.ravel()])

## 岭回归分类目标函数计算
# def rls(W, X, Y, lmbd):
# n_samples, n_features = X.shape
# _, n_classes = Y.shape
## 从向量形式的优化变量中还原W和b
# b = W[-n_classes:]
# W = W[:-n_classes].reshape(n_features, n_classes)

# r = np.sum((X@W + b - Y) ** 2) / (2 * n_samples) +  lmbd * np.sum(W ** 2)
# return r

## 岭回归分类梯度计算
# def rls_der(W, X, Y, lmbd):
# n_samples, n_features = X.shape
# _, n_classes = Y.shape
# b = W[-n_classes:]
# W = W[:-n_classes].reshape(n_features, n_classes)

## 分别计算W和b的梯度
# der_W = ((X.T@X)@W + np.sum(X, 0).reshape(n_features, 1)@(b.reshape(1, n_classes)) - X.T@Y) / n_samples + 2 * lmbd * W
# der_b = b + np.mean(X, 0)@W - np.mean(Y, 0)
## 将W和b的梯度按照向量形式合并
# der = np.concatenate([der_W.ravel(), der_b.ravel()])
# return der

## 定义回调函数，记录每次优化后的目标函数值
# train_loss_list = []
# test_loss_list = []
# def callbackF(W):
# global train_loss_list, test_loss_list
# b = W[-n_classes:]
# W = W[:-n_classes].reshape(n_features, n_classes)
# rls_value = np.sum((X_train@W + b - y_train_coding) ** 2) / (2 * n_samples) +  lmbd * np.sum(W ** 2)
# train_loss_list.append(rls_value)
# rls_value = np.sum((X_test@W + b - y_test_coding) ** 2) / (2 * n_samples) +  lmbd * np.sum(W ** 2)
# test_loss_list.append(rls_value)

# lmbd = 0.01
## 调用优化函数，传入目标值、梯度的计算函数，选择BFGS为优化方法
# res = minimize(rls, m0, args=(X_train, y_train_coding, lmbd), method='BFGS', jac=rls_der,
# options={'gtol': 1e-6, 'disp': True}, callback=callbackF)
# m = res.x                                               # 获得最终优化变量
# b = m[-n_classes:]
# W = m[:-n_classes].reshape(n_features, n_classes)
# y_pred = np.argmax(X_test@W + b, 1)
# acc_rls_gd = accuracy_score(y_test, y_pred)
# print('RLS with gradient descent solver gets accuracy {} over {} iterations'.format(acc_rls_gd, res.nit))

# import matplotlib
# from matplotlib import pyplot as plt
# config = {
# 'font.family': 'serif',
# 'font.serif': ['simHei'],
# 'font.sans-serif': ['Times New Roman'],
# 'axes.unicode_minus': False,
# 'mathtext.fontset': 'cm',
# 'font.size': 18,
# }
# plt.rcParams.update(config)

# matplotlib.rcParams['figure.figsize'] = [8, 6]
# matplotlib.rc('axes', linewidth=2)
# matplotlib.rcParams['font.sans-serif'] = ['SimSun']
# matplotlib.rcParams['axes.unicode_minus']=False
## plt.title("MSE-$\lambda$")
# plt.xlabel(u"迭代轮数", size=22)
# plt.ylabel("MSE", size=22)
# plt.plot(train_loss_list, 'r', linewidth = 5, label=u"训练集损失")
# plt.plot(test_loss_list, 'b--', linewidth = 5, label=u"测试集损失")
# plt.ylim([0,40])
# plt.legend()
## plt.show()
# plt.savefig('gd_res.pdf')

'''
-----------------------------------------------------------
------------------------   11.3   -------------------------
-----------------------------------------------------------
'''
### implement (stochastic gradeint descent) least square solver
from tqdm import tqdm

# SGD的超参数
batch_size = 128
eta_init = 0.001  # default 0.1
max_iter = 1000
lmbd = 0.001
val_interval = 100
# 初始化模型参数
W = np.random.rand(n_features, n_classes)
b = np.zeros((n_classes))
# 用于记录统计量
train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []


# 目标函数
def rls_sgd(W, b, X, Y, lmbd):
    r = np.sum((X @ W + b - Y) ** 2) / (2 * n_samples) + lmbd * np.sum(W ** 2)
    return r


# 随机梯度计算
def rls_der_sgd(W, b, X, Y, lmbd):
    n_samples, n_features = X.shape
    _, n_classes = Y.shape
    XT1 = np.sum(X, 0).reshape(n_features, 1)
    # 随机梯度计算方法和梯度计算方法一致
    der_W = ((X.T @ X) @ W + XT1 @ (b.reshape(1, n_classes)) - (X.T) @ Y) / n_samples + 2 * lmbd * W
    der_b = b + np.mean(X, 0) @ W - np.mean(Y, 0)

    return der_W, der_b


# 进行优化迭代
for i in tqdm(range(max_iter + 1)):
    eta = eta_init  # 可选：学习率逐轮下降，令eta = eta_init / (i + 1)
    # 每迭代一定轮数后计算整个训练、测试集的目标函数和准确率
    if i % val_interval == 0:
        # 计算训练集统计量
        train_loss = rls_sgd(W, b, X_train, y_train_coding, lmbd)
        train_acc = accuracy_score(y_train, np.argmax(X_train @ W + b, 1))
        # 计算测试集统计量
        test_loss = rls_sgd(W, b, X_test, y_test_coding, lmbd)
        test_acc = accuracy_score(y_test, np.argmax(X_test @ W + b, 1))
        # 记录统计量
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc * 100)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc * 100)
    # 随机抽取batch_size个样本
    sample_ids = np.random.choice(X_train.shape[0], batch_size)
    X_train_batch, y_train_coding_batch = X_train[sample_ids, :], y_train_coding[sample_ids]
    # 计算随机梯度
    der_W, der_b = rls_der_sgd(W, b, X_train_batch, y_train_coding_batch, lmbd)
    # 利用随机梯度实现更新
    W = W - eta * der_W
    b = b - eta * der_b

y_pred = np.argmax(X_test @ W + b, 1)
acc_rls_sgd = accuracy_score(y_test, y_pred)
print('RLS with stochastic gradient descent solver gets accuracy {}'.format(acc_rls_sgd))

import matplotlib
from matplotlib import pyplot as plt

config = {
    'font.family': 'serif',
    'font.serif': ['simHei'],
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
    'font.size': 18,
}
plt.rcParams.update(config)

# plt.clf()
# matplotlib.rcParams['figure.figsize'] = [8, 6]
# matplotlib.rc('axes', linewidth=2)
# matplotlib.rcParams['font.sans-serif'] = ['SimSun']
# matplotlib.rcParams['axes.unicode_minus']=False
## plt.title("MSE-$\lambda$")
# plt.xlabel(u"迭代轮数", size=22)
# plt.ylabel("MSE", size=22)
# plt.xticks([0, 2, 4, 6, 8, 10], ['0', '200', '400', '600', '800', '1000'])
# plt.plot(train_loss_list, 'r', linewidth = 5, label=u"训练集损失")
# plt.plot(test_loss_list, 'b--', linewidth = 5, label=u"测试集损失")
# plt.ylim([0, 10])
# plt.legend()
## plt.show()
# plt.savefig('sgd_res_loss.pdf')

plt.figure(figsize=(8, 6), dpi=80)
plt.clf()
plt.xlabel(u"迭代轮数", size=22)
plt.ylabel(u"精度", size=22)
plt.xticks([0, 2, 4, 6, 8, 10], ['0', '200', '400', '600', '800', '1000'])
plt.plot(train_acc_list, 'r', linewidth=5, label=u"训练集精度")
plt.plot(test_acc_list, 'b--', linewidth=5, label=u"测试集精度")
plt.ylim([0, 100])
plt.legend()
# plt.show()
plt.savefig('ch11_sgd_res_acc.pdf')

'''
-----------------------------------------------------------
------------------------   11.4   -------------------------
-----------------------------------------------------------
'''
### use SVM as a reference
# from sklearn.model_selection import KFold
# from sklearn.utils.fixes import loguniform
# from sklearn.model_selection import GridSearchCV

# kf = KFold(n_splits=5)
# C_list = np.logspace(-5, 5, 11)
# cv_accuracy = np.zeros((5, len(C_list)))
# for fold_index, (train_index, test_index) in enumerate(kf.split(X_train)):
# X_train_train, X_train_val = X_train[train_index], X_train[test_index]
# y_train_train, y_train_val = y_train[train_index], y_train[test_index]
# for c_index, c_value in enumerate(C_list):
# clf = LinearSVC(random_state=0, tol=1e-5, C = c_value, dual=False, max_iter=10000)
# clf.fit(X_train_train, y_train_train)
# y_pred = clf.predict(X_train_val)
# cv_accuracy[fold_index, c_index] = accuracy_score(y_train_val, y_pred)
# best_c = C_list[np.argmax(np.mean(cv_accuracy, 0))]
# clf = LinearSVC(random_state=0, tol=1e-5, C=best_c, dual=False, max_iter=10000)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# svm_acc = accuracy_score(y_test, y_pred)
# print('SVM with Grid Search on C gets accuracy {}'.format(svm_acc))

## param_grid = {
##     "C": np.logspace(-5, 5, 11)
## }
## clf = GridSearchCV(
##     LinearSVC(random_state=0, tol=1e-5, dual=False, max_iter=10000), param_grid,
## )
## clf = clf.fit(X_train, y_train)
## y_pred = clf.predict(X_test)
## svm_acc2 = accuracy_score(y_test, y_pred)
## print('SVM with GridSearchCV on C gets accuracy {}'.format(svm_acc2))

### use RLS as a reference
# from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV

# kf = KFold(n_splits=5)                              # 产生K个均匀的数据划分
# alpha_list = np.logspace(-5, 5, 11)                 # 设置待选参数
# cv_accuracy = np.zeros((5, len(alpha_list)))
# for fold_index, (train_index, test_index) in enumerate(kf.split(X_train)):
# X_train_train, X_train_val = X_train[train_index], X_train[test_index]  # 获得每一个划分的训练集和验证集样例
# y_train_train, y_train_val = y_train[train_index], y_train[test_index]  # 获得每一个划分的训练集和验证集标记
# for alpha_index, alpha_value in enumerate(alpha_list):
# clf = RidgeClassifier(random_state=0, alpha = alpha_value)          # 尝试不同的超参数进行模型训练
# clf.fit(X_train_train, y_train_train)
# y_pred = clf.predict(X_train_val)
# cv_accuracy[fold_index, alpha_index] = accuracy_score(y_train_val, y_pred)
# best_alpha = alpha_list[np.argmax(np.mean(cv_accuracy, 0))]                 # 选择最优性能对应的超参数
# clf = RidgeClassifier(random_state=0, alpha=best_alpha)                     # 使用最优超参数在完整的训练集上训练模型
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# rls_acc = accuracy_score(y_test, y_pred)
# print('RLS with Grid Search on C gets accuracy {}'.format(rls_acc))

## 通过字典定义需要选择的超参数以及其范围
# param_grid = {"alpha": np.logspace(-5, 5, 11)}
## 构建能够自动进行参数选择的分类器
# clf = GridSearchCV(
# RidgeClassifier(random_state=0), param_grid,
# )
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# rls_acc2 = accuracy_score(y_test, y_pred)
# print('RLS with GridSearchCV on C gets accuracy {}'.format(rls_acc2))

## for lmbd in np.logspace(-5, 5, 11):
##     W, b = ridgeReg(X_train, y_train_coding, lmbd)
##     y_pred = np.argmax(X_test@W + b, 1)
##     acc_rls = accuracy_score(y_test, y_pred)
##     print('RLS with $lambda$ = {} closed-form solver gets accuracy {}'.format(lmbd, acc_rls))


'''
-----------------------------------------------------------
------------------------   11.5   -------------------------
-----------------------------------------------------------
'''


# 实现岭回归分类器的闭式解求解函数
def ridgeReg(X, Y, lmbd):
    n_samples, n_features = X.shape
    # 构建中心化矩阵H
    H = np.identity(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    HX = H @ X
    # 基于闭式解公式求解W和b
    W = np.linalg.pinv(HX.T @ HX / n_samples + 2 * lmbd * np.identity(n_features)) @ (HX.T @ Y) / n_samples
    b = np.mean(Y - (X @ W), 0)
    return W, b


np.random.seed(0)
# 对训练集进一步划分，模拟小样本训练集场景
X_train_many, X_train_few, y_train_many, y_train_few = train_test_split(
    X_train, y_train, test_size=0.05, random_state=100
)
assert (np.unique(y_train_many).size == np.unique(y_train_few).size)
print('There are only {} instances in the target task'.format(y_train_few.shape[0]))
# 在训练数据上增加噪声，以模拟分布的变化
X_train_many = X_train_many + 2 * np.random.randn(*X_train_many.shape)
lmbd = 0.01
y_train_many_coding = - np.ones((y_train_many.size, n_classes))
y_train_many_coding[np.arange(y_train_many.size), y_train_many] = 1
W, b = ridgeReg(X_train_many, y_train_many_coding, lmbd)
y_pred = np.argmax(X_test @ W + b, 1)
acc_rls_many = accuracy_score(y_test, y_pred)
W_many, b_many = W, b
print('RLS trained from the many-shot part accuracy {}'.format(acc_rls_many))

# lmbd = 0.01
y_train_few_coding = - np.ones((y_train_few.size, n_classes))
y_train_few_coding[np.arange(y_train_few.size), y_train_few] = 1
# W, b = ridgeReg(X_train_few, y_train_few_coding, lmbd)
# y_pred = np.argmax(X_test@W + b, 1)
# acc_rls_few = accuracy_score(y_test, y_pred)
# print('RLS trained from the few-shot part accuracy {}'.format(acc_rls_few))

# def ridgeRegBiased(X, Y, lmbd, W_many):
# n_samples, n_features = X.shape
## 构建中心化矩阵H
# H = np.identity(n_samples) - np.ones((n_samples, n_samples)) / n_samples
# HX = H @ X
## 基于闭式解公式求解W和b
# W = np.linalg.pinv(HX.T @ HX / n_samples  + 2 * lmbd * np.identity(n_features)) @ (HX.T @ Y / n_samples + 2 * lmbd * W_many)
# b = np.mean(Y - (X@W), 0)
# return W, b

# lmbd = 0.01
# W, b = ridgeRegBiased(X_train_few, y_train_few_coding, lmbd, W_many)
# y_pred = np.argmax(X_test@W + b, 1)
# acc_rls_reuse = accuracy_score(y_test, y_pred)
# print('RLS trained on few-shot part with a pre-trained classifier get accuracy {}'.format(acc_rls_reuse))


# SGD用于biased regularization
from tqdm import tqdm

eta_init = 0.01
max_iter = 100
lmbd = 0.1
val_interval = 100
# 使用相关任务模型参数初始化当前模型
W, b = W_many, b_many


# 随机梯度计算
def rls_der_sgd(W, b, X, Y, lmbd, W0, b0):
    n_samples, n_features = X.shape
    _, n_classes = Y.shape
    XT1 = np.sum(X, 0).reshape(n_features, 1)
    # 考虑相关任务模型的影响
    der_W = ((X.T @ X) @ W + XT1 @ (b.reshape(1, n_classes)) - (X.T) @ Y) / n_samples + 2 * lmbd * (W - W0)
    der_b = b + np.mean(X, 0) @ W - np.mean(Y, 0) + 2 * lmbd * (b - b0)

    return der_W, der_b


# 进行优化迭代
for i in tqdm(range(max_iter + 1)):
    eta = eta_init
    # 计算随机梯度
    der_W, der_b = rls_der_sgd(W, b, X_train_few, y_train_few_coding, lmbd, W_many, b_many)
    # 利用随机梯度实现更新
    W = W - eta * der_W
    b = b - eta * der_b

y_pred = np.argmax(X_test @ W + b, 1)
acc_rls_sgd = accuracy_score(y_test, y_pred)
print('Biased RLS with stochastic gradient descent solver gets accuracy {}'.format(acc_rls_sgd))

'''
-----------------------------------------------------------
------------------------  others  -------------------------
-----------------------------------------------------------
'''
# def rls_biased(W, X, Y, lmbd, W0):
# n_samples, d = X.shape
# _, n_classes = Y.shape
# b = W[-n_classes:]
# W = W[:-n_classes].reshape(d, n_classes)

# r = np.sum((X@W + b - Y) ** 2) / 2 + lmbd * np.sum((W - W0) ** 2)
# return r

# def rls_der_biased(W, X, Y, lmbd, W0):
# n_samples, d = X.shape
# _, n_classes = Y.shape
# b = W[-n_classes:]
# W = W[:-n_classes].reshape(d, n_classes)

# der_W = (X.T@X)@W + np.sum(X, 0).reshape(d, 1)@(b.reshape(1, n_classes)) - X.T@Y + 2 * lmbd * (W - W0)
# der_b = np.sum(X, 0)@W + n_samples * b - np.sum(Y, 0)

# der = np.concatenate([der_W.ravel(), der_b.ravel()])
# return der

# lmbd = 10
# m0 = np.concatenate([W_many.ravel(), b_many.ravel()])
# y_train_few_coding = - np.ones((y_train_few.size, n_classes))
# y_train_few_coding[np.arange(y_train_few.size), y_train_few] = 1
# res = minimize(rls_biased, m0, args=(X_train_few, y_train_few_coding, lmbd, W_many), method='BFGS', jac=rls_der_biased,
# options={'gtol': 1e-6, 'disp': True})
# m = res.x
# b = m[-n_classes:]
# W = m[:-n_classes].reshape(n_components, n_classes)
# y_pred = np.argmax(X_test@W + b, 1)
# acc_rls_reuse = accuracy_score(y_test, y_pred)
# print('RLS trained on few-shot part with a pre-trained classifier get accuracy {}'.format(acc_rls_reuse))


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# clf = LinearSVC(random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# svm_acc = accuracy_score(y_test, y_pred)
# print('After StandardScaler, SVM gets accuracy {}'.format(svm_acc))

# clf = RidgeClassifier(random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# rls_acc = accuracy_score(y_test, y_pred)
# print('After StandardScaler, RLS gets accuracy {}'.format(rls_acc))
