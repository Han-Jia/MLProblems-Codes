# -*- coding: utf-8 -*-
"""
@Author Su Lu

@Date: 2022-02-18 17:20:28
"""

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'


def cal_gauss(X, lambd):
    Y = np.exp(-1 * (X ** 2) / (2 * lambd)) / np.sqrt(2 * np.pi * lambd)
    return Y


def cal_laplace(X, lambd):
    Y = np.exp(-np.abs(X) / lambd) / (2 * lambd)
    return Y


def draw_distribution():
    w = 5
    X = np.linspace(-w, w, 500)
    lambd_gauss = 1
    lambd_laplace = 1
    Y1 = cal_gauss(X, lambd_gauss)
    Y2 = cal_laplace(X, lambd_laplace)

    fig = plt.figure(figsize=(20, 15))

    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('$x$', fontsize=56)
    ax.set_xlim(-w - 0.2, w + 0.2)
    ax.set_xticks([-w, 0, w])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': 56})
    ax.set_ylabel('$p(x)$', fontsize=56)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, 0.5])
    ax.set_yticklabels([0, 0.5], fontdict={'fontsize': 56})

    ax.spines['top'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)

    ax.plot(X, Y1, linewidth=5, color='#C96C44', label='Gaussian')
    ax.plot(X, Y2, linewidth=5, color='#47BE55', label='Laplacian')

    ax.legend(prop={'size': 56})

    plt.show()


if __name__ == '__main__':
    draw_distribution()
