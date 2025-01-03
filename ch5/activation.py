# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

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

plt.clf()
fig = plt.figure(figsize=(7, 5), dpi=80)
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-2, 2, 100)
plt.plot(x, 1 / (1 + np.exp(-5 * x)), "#A52A2A", linewidth=3)
plt.xticks(np.arange(min(x), max(x) + 1, 1.0), font='sans serif')
plt.yticks([-2, -1, 0, 1, 2], font='sans serif')
ax.grid(True)
plt.savefig('sigmoid.pdf')

plt.clf()
fig = plt.figure(figsize=(7, 5), dpi=80)
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-2, 2, 100)
plt.plot(x, (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), "#A52A2A", linewidth=3)
plt.xticks(np.arange(min(x), max(x) + 1, 1.0), font='sans serif')
plt.yticks([-2, -1, 0, 1, 2], font='sans serif')
ax.grid(True)
plt.savefig('tanh.pdf')

plt.clf()
fig = plt.figure(figsize=(7, 5), dpi=80)
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-2, 2, 100)
plt.plot(x, np.maximum(x, 0), "#A52A2A", linewidth=3)
plt.xticks(np.arange(min(x), max(x) + 1, 1.0), font='sans serif')
plt.yticks([-2, -1, 0, 1, 2], font='sans serif')
ax.grid(True)
plt.savefig('relu.pdf')

plt.clf()
fig = plt.figure(figsize=(7, 5), dpi=80)
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-2, 2, 100)
plt.plot(x, np.log(1 + np.exp(x)), "#A52A2A", linewidth=3)
plt.xticks(np.arange(min(x), max(x) + 1, 1.0), font='sans serif')
plt.yticks([-2, -1, 0, 1, 2], font='sans serif')
ax.grid(True)
plt.savefig('softplus.pdf')

plt.clf()
fig = plt.figure(figsize=(7, 5), dpi=80)
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-2, 0, 50)
plt.plot(x, 0.1 * x, "#A52A2A", linewidth=3)
x = np.linspace(0, 2, 50)
plt.plot(x, x, "#A52A2A", linewidth=3)
plt.xticks(np.arange(-2, 2 + 1, 1.0), font='sans serif')
plt.yticks([-2, -1, 0, 1, 2], font='sans serif')
ax.grid(True)
plt.savefig('leaky.pdf')

plt.clf()
fig = plt.figure(figsize=(7, 5), dpi=80)
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-2, -0.05, 50)
plt.plot(x, np.zeros_like(x), "#A52A2A", linewidth=3)
x = np.linspace(0.05, 2, 50)
plt.plot(x, np.ones_like(x), "#A52A2A", linewidth=3)
plt.scatter(0, 0, marker='o', s=30, facecolors='none', edgecolors="#A52A2A", )
plt.xticks(np.arange(-2, 2 + 1, 1.0), font='sans serif')
plt.yticks([-2, -1, 0, 1, 2], font='sans serif')
ax.grid(True)
plt.savefig('zeroone.pdf')

print('finish')
