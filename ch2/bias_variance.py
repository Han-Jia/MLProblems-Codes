#!/usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib import font_manager
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
and https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/bias_variance_decomp.py
'''

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 20}

matplotlib.rc('font', **font)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rc('axes', linewidth=2)

font_p = font_manager.FontProperties()
font_p.set_family('SimHei')
font_p.set_size(14)


def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])          # 获取输入数据大小m
    # 从m个样本中有放回抽样m次, 构成同样大小的数据集
    bootstrap_indices = rng.choice(sample_indices,
                                   size=sample_indices.shape[0],
                                   replace=True)
    return X[bootstrap_indices], y[bootstrap_indices]


def bias_variance_decomp(estimator, X_train, y_train, X_test, y_test,
                         num_rounds=200, random_seed=None, returned_num=3,
                         **fit_params):
    rng = np.random.RandomState(random_seed)
    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=np.float64)
    estimator_list = []

    # 通过num_rounds次自助法抽样模拟不同的训练集
    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)
        # 拟合自助法构成的训练数据, Pipeline的拟合和预测需要二维矩阵输入
        sampled_estimator = estimator.fit(
            X_boot[:, np.newaxis], y_boot, **fit_params)
        pred = sampled_estimator.predict(X_test[:, np.newaxis])
        estimator_list.append(copy.deepcopy(sampled_estimator))
        all_pred[i] = pred

    # 估计平方损失的期望
    avg_expected_loss = np.apply_along_axis(
        lambda x:
        ((x - y_test) ** 2).mean(),
        axis=1,
        arr=all_pred).mean()
    main_predictions = np.mean(all_pred, axis=0)

    avg_bias = np.sum((main_predictions - y_test) ** 2) / y_test.size  # 估计偏差
    avg_var = np.sum((main_predictions - all_pred) ** 2) / all_pred.size  # 估计方差

    returned_model = [estimator_list[e]
                      for e in np.random.choice(len(estimator_list), returned_num)]
    return avg_expected_loss, avg_bias, avg_var, returned_model


# 定义真实曲线（x, y的关系）
def true_fun(X):
    return np.cos(1.5 * np.pi * X + 0.5 * np.pi)


np.random.seed(0)
n_samples = 100  # 共100个样本
degrees = [1, 4, 30]
X = np.sort(np.random.rand(n_samples))  # 随机抽取100个x
y = true_fun(X) + np.random.randn(n_samples) * 0.1  # 根据真实曲线加上噪声产生y
# 进行训练集和测试集的划分，各占总数据量的50%
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=123,
                                                    shuffle=True)

matplotlib.rc('axes', linewidth=2)

for i in range(len(degrees)):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    plt.setp(ax, xticks=(), yticks=())

    # 产生degree阶多项式特征
    polynomial_features = PolynomialFeatures(
        degree=degrees[i], include_bias=False)
    # 构建线性回归模型
    linear_regression = LinearRegression()
    # 针对多项式特征构建线性回归模型
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )

    avg_expected_loss, avg_bias, avg_var, returned_model = bias_variance_decomp(
        pipeline, X_train, y_train, X_test, y_test,
        random_seed=123)

    print('Average expected loss: %.3f' % avg_expected_loss)
    print('Average bias: %.3f' % avg_bias)
    print('Average variance: %.3f' % avg_var)

    X_plot = np.linspace(0, 1, 100)
    color = ['b', 'r', 'g']
    plt.scatter(X_train, y_train, c="k", s=30, label=u"训练数据")
    for j, sampled_model in enumerate(returned_model):
        plt.plot(X_plot, sampled_model.predict(X_plot[:, np.newaxis]),
                 color[j], linewidth=2, label=u"模型-{}".format(j + 1))

    plt.xlim((0, 1))
    plt.ylim((-2, 2))

    plt.legend(loc="best", prop=font_p)
    # plt.plot(X_test, true_fun(X_test), label="True function")
    plt.savefig('ch2_{}-degree.pdf'.format(degrees[i]))
