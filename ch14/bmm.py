# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2022-05-05 13:13:33
"""

import copy
import numpy as np


# 函数bernoulli_mixture计算伯努利混合分布在某点$x$处的概率密度
def bernoulli_mixture(K: int, pi: np.ndarray, theta: np.ndarray, x: int):
    # K: 伯努利混合成分的个数
    # pi: 各成分的权重
    # theta: 各成分模型参数
    # x: 观测样本点

    p1 = np.power(theta, x)
    p2 = np.power(1 - theta, 1 - x)
    p = np.sum(pi * p1 * p2)
    return p


# 函数data_likelihood计算观测数据的似然
def data_log_likelihood(K: int, pi: np.ndarray, theta: np.ndarray, bx: np.ndarray):
    # K: 伯努利混合成分的个数
    # pi: 各成分的权重
    # theta: 各成分模型参数
    # bx: 观测数据集

    log_likelihood = 0.0
    for i in range(0, len(bx)):
        log_likelihood = log_likelihood + np.log(bernoulli_mixture(K, pi, theta, bx[i]))

    return log_likelihood


# 函数iter\_em实现了第一小问中的迭代优化算法
def iter_em(K, bx):
    # K: 伯努利混合成分的个数
    # bx: 观测数据集

    # 初始化参数
    pi = np.random.rand(K)
    pi = pi / np.sum(pi)
    theta = np.random.rand(K)

    print('initial log_likelihood: ', data_log_likelihood(K, pi, theta, bx))

    converge = False
    iter = 0
    while not converge:
        # 计算变量$\{\gamma_{ik}\}$
        gamma = np.zeros((len(bx), K))
        for i in range(0, len(bx)):
            px_i = bernoulli_mixture(K, pi, theta, bx[i])
            for k in range(0, K):
                gamma[i][k] = pi[k] * np.power(theta[k], bx[i]) * np.power(1 - theta[k], 1 - bx[i])
                gamma[i][k] = gamma[i][k] / px_i

        # 计算参数$\{\theta_{k}, \pi_{k}\}$
        new_theta = np.zeros_like(theta)
        new_pi = np.zeros_like(pi)
        for k in range(0, K):
            new_theta[k] = np.sum(gamma[:, k] * bx.T) / np.sum(gamma[:, k])
            new_pi[k] = np.sum(gamma[:, k]) / len(bx)

        iter += 1
        print('iter %d finish, log_likelihood = %f' % (iter, data_log_likelihood(K, new_pi, new_theta, bx)))

        # 判断是否收敛
        if np.sum(np.abs(new_theta - theta)) < 1e-6 and np.sum(np.abs(new_pi - pi)) < 1e-6:
            converge = True
        else:
            theta = copy.copy(new_theta)
            pi = copy.copy(new_pi)

    return theta, pi, gamma


if __name__ == '__main__':
    # np.random.seed(648)

    K = 10
    bx = np.random.randint(0, 2, size=100)

    theta, pi, gamma = iter_em(K, bx)
    # print('weights of components:')
    # print(pi)
    # print('parameters of components:')
    # print(theta)

    # for i in range(0, len(bx)):
    #     print('data %d: %d' % (i, bx[i]))
    #     print('posterior distribution:')
    #     print(gamma[i])
