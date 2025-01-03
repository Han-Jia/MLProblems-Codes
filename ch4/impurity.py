# -*- coding: utf-8 -*-
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

x = np.linspace(0, 1, 100)
color = ['b', 'r', 'g']
# 误分类错误率
plt.plot(x, 1 - np.maximum(x, 1 - x), 'b', linewidth=2, label=u"误分类错误率")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
plt.xlabel('$p_k$')
plt.ylabel(u'不纯度')
# plt.legend(loc="best", prop=fontP)
# plt.plot(X_test, true_fun(X_test), label="True function")
plt.savefig('ch4_misclassification.pdf')

plt.clf()
x = np.linspace(0, 1, 100)
color = ['b', 'r', 'g']
# Gini
plt.plot(x, 2 * x * (1 - x), 'r', linewidth=2, label=u"基尼指数")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
plt.xlabel('$p_k$')
plt.ylabel(u'不纯度')
# plt.legend(loc="best", prop=fontP)
# plt.plot(X_test, true_fun(X_test), label="True function")
plt.savefig('ch4_gini.pdf')

plt.clf()
x = np.linspace(0, 1, 100)
color = ['b', 'r', 'g']
# Entropy
plt.plot(x, - x * np.log2(x) - (1 - x) * np.log2(1 - x), 'g', linewidth=2, label=u"信息熵")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
plt.xlabel('$p_k$')
plt.ylabel(u'不纯度')
# plt.legend(loc="best", prop=fontP)
# plt.plot(X_test, true_fun(X_test), label="True function")
plt.savefig('ch4_entropy.pdf')
