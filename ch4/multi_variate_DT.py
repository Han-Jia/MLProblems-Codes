# -*- coding: utf-8 -*-

import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

config = {
    'font.family': 'serif',
    'font.serif': ['kaiti'],
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
    'font.size': 18,
}
plt.rcParams.update(config)

matplotlib.rc('axes', linewidth=2)


def single_variate_DT(x_1, x_2):
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    for a, b in zip(x_1, x_2):
        if b < 30:
            neg_x.append(a)
            neg_y.append(b)
        else:
            if b > 60:
                pos_x.append(a)
                pos_y.append(b)
            else:
                if a < 40:
                    pos_x.append(a)
                    pos_y.append(b)
                else:
                    neg_x.append(a)
                    neg_y.append(b)
    return pos_x, pos_y, neg_x, neg_y


def multi_variate_DT(x_1, x_2):
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    for a, b in zip(x_1, x_2):
        if 2 * a / 45 - 7 * b / 90 - 1 >= 0:
            neg_x.append(a)
            neg_y.append(b)
        else:
            pos_x.append(a)
            pos_y.append(b)
    return pos_x, pos_y, neg_x, neg_y


label_1_x1 = [24, 25, 32, 52, 22, 48]
label_1_x2 = [40, 77, 48, 110, 38, 65]
label_0_x1 = [53, 23, 43, 52]
label_0_x2 = [52, 25, 44, 27]

# pos_x, pos_y, neg_x, neg_y = single_variate_DT(x_1, x_2)
plt.figure(figsize=(7, 7), dpi=80)
plt.scatter(label_1_x1, label_1_x2, s=100, c='b', label=u'标记$=1$', marker='+')
plt.scatter(label_0_x1, label_0_x2, s=100, c='r', label=u'标记$=0$', marker='_')
horizontal_1 = np.linspace(20, 40, 100)
horizontal_2 = np.linspace(40, 56, 100)
vertical_1 = np.linspace(30, 60, 100)
plt.plot(horizontal_1, [30] * 100, "#A52A2A", linewidth=3, label=u'决策边界')
plt.plot([40] * 100, vertical_1, "#A52A2A", linewidth=3)
plt.plot(horizontal_2, [60] * 100, "#A52A2A", linewidth=3)
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
plt.xlabel(u'属性$a_1$')
plt.ylabel(u'属性$a_2$')
plt.legend()
plt.savefig('ch4_single_variate_DT_mp.pdf')

plt.clf()
plt.figure(figsize=(7, 7), dpi=80)
# pos_x, pos_y, neg_x, neg_y = multi_variate_DT(x_1, x_2)
plt.scatter(label_1_x1, label_1_x2, s=100, c='b', label=u'标记$=1$', marker='+')
plt.scatter(label_0_x1, label_0_x2, s=100, c='r', label=u'标记$=0$', marker='_')
horizontal = np.linspace(20, 60, 100)
plt.plot(horizontal, horizontal + 10, "#20B2AA", linewidth=3, label=u'$-1/10a_1+1/10a_2-1=0$')
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
plt.xlabel(u'属性$a_1$')
plt.ylabel(u'属性$a_2$')
plt.legend()
plt.savefig('ch4_multi_variate_DT_mp.pdf')
