# -*- coding: utf-8 -*-
"""
@Author Su Lu

@Date: 2022-02-15 17:00:15
"""

import numpy as np
from scipy.special import gamma
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'


def draw_likelihood(N_1, N_0):
    fig = plt.figure(figsize=(20, 15))

    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('$\\theta$', fontsize=56)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontdict={'fontsize': 56})
    ax.set_ylabel('Bernoulli likelihood', fontsize=56)
    ax.set_ylim(0, 0.2)
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_yticklabels([0, 0.1, 0.2], fontdict={'fontsize': 56})

    ax.spines['top'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)

    X = np.linspace(0, 1, 100)
    Y = np.power(X, N_1) * np.power(1 - X, N_0)

    ax.plot(X, Y, linewidth=10, color='#47BE55', label='probability')
    text = '$N_1$ = ' + str(N_1) + '\n' + \
           '$N_0$ = ' + str(N_0)
    ax.text(0.05, 0.78, text, fontsize=56)

    mle = N_1 / (N_1 + N_0)
    ax.vlines(x=mle, ymin=0, ymax=2, linewidth=5, color='#8269C0', linestyle='dashed', label='MLE')

    ax.legend(prop={'size': 56})

    plt.show()


def draw_beta_distribution(a, b):
    fig = plt.figure(figsize=(20, 15))

    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('$\\theta$', fontsize=56)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontdict={'fontsize': 56})
    ax.set_ylabel('probability density', fontsize=56)
    ax.set_ylim(0, 3.2)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([0, 1, 2, 3], fontdict={'fontsize': 56})

    ax.spines['top'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)

    X = np.linspace(0, 1, 100)
    c = gamma(a + b) / (gamma(a) * gamma(b))
    Y = c * np.power(X, a - 1) * np.power(1 - X, b - 1)

    ax.plot(X, Y, linewidth=10, color='#C96C44', label='probability')
    text = 'prior' + '\n' + \
           'a = ' + str(a) + '\n' + \
           'b = ' + str(b)
    # text = 'posterior' + '\n' + \
    #        'a = ' + str(a) + '\n' + \
    #        'b = ' + str(b)
    ax.text(0.05, 2.3, text, fontsize=56)

    # if a + b - 2 != 0:
    #     mode = (a - 1) / (a + b - 2)
    #     ax.vlines(x=mode, ymin=0, ymax=5, linewidth=5, color='#FBC40F', linestyle='dashed', label='mode')

    # N_1, N_0 = a - 1, b - 2
    # cum_mle = N_1 / (N_1 + N_0)
    # ax.vlines(x=cum_mle, ymin=0, ymax=5, linewidth=5, color='#8269C0', linestyle='dashed', label='cumulated MLE')

    # ax.legend(prop={'size': 42}, loc='lower left')
    ax.legend(prop={'size': 42}, loc='upper right')

    plt.show()


if __name__ == '__main__':
    N_1, N_0 = 3, 1
    draw_likelihood(N_1, N_0)
    # a, b = 6, 5
    # draw_beta_distribution(a, b)
