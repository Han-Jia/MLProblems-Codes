# @author : Administrator 
# @date : 2022/4/20
import os.path

import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

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

bg_rgb = matplotlib.colors.ListedColormap([
    # '#F5DEB3','#BDFCC9',
    '#B0E0E6', '#FFC0CB'])
data_rgb = matplotlib.colors.ListedColormap([
    # '#FFD700','#228B22',
    '#1E90FF', '#E3170D'])

# X, y = make_circles(n_samples=1000, factor=0.5, noise=0.2, random_state=0)
X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

margin = 0.3
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
# fig = plt.figure()

# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired, edgecolors="k")
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap=plt.cm.Paired, edgecolors="k", alpha=0.75,
#             linestyle='dashed')
# plt.ylabel("Feature #1")
# plt.xlabel("Feature #0")
# plt.title("Training/Test data")
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.savefig(os.path.join('figures', 'two_moons.pdf'), transparent=True, bbox_inches='tight')
# plt.close(fig)

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

os.makedirs('figures', exist_ok=True)

h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=bg_rgb, alpha=1.)
markers = ['+', '_']
for i in range(2):
    idx = y_train == i
    plt.scatter(X_train[idx][:, 0], X_train[idx][:, 1], c=data_rgb.colors[i], s=30, marker=markers[i])
    idx = y_test == i
    plt.scatter(X_test[idx][:, 0], X_test[idx][:, 1], c=data_rgb.colors[i], s=30, marker=markers[i], alpha=0.75,
                linestyle='dashed')
# plt.ylabel("测试精度 #1")
# plt.xlabel("Feature #0")
plt.xticks([])
plt.yticks([])
plt.title(f"k-Nearest Neighbor")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.text(
    xx.max() - 0.1,
    yy.min() + 0.1,
    ("测试精度 : %.2f" % score).lstrip("0"),
    size=14,
    horizontalalignment="right",
)
plt.savefig(os.path.join('figures', f'ch16_two_moons_knn.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)

clf = LogisticRegression()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=bg_rgb, alpha=1.)
for i in range(2):
    idx = y_train == i
    plt.scatter(X_train[idx][:, 0], X_train[idx][:, 1], c=data_rgb.colors[i], s=30, marker=markers[i])
    idx = y_test == i
    plt.scatter(X_test[idx][:, 0], X_test[idx][:, 1], c=data_rgb.colors[i], s=30, marker=markers[i], alpha=0.75,
                linestyle='dashed')
# plt.ylabel("Feature #1")
# plt.xlabel("Feature #0")
plt.xticks([])
plt.yticks([])
plt.title(f"Logistic Regression")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.text(
    xx.max() - 0.1,
    yy.min() + 0.1,
    ("测试精度 : %.2f" % score).lstrip("0"),
    size=14,
    horizontalalignment="right",
)
plt.savefig(os.path.join('figures', f'ch16_two_moons_lr.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)

for K in [1, 5, 129]:
    clf = KNeighborsClassifier(K)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=bg_rgb, alpha=1.)
    for i in range(2):
        idx = y_train == i
        plt.scatter(X_train[idx][:, 0], X_train[idx][:, 1], c=data_rgb.colors[i], s=30, marker=markers[i])
        idx = y_test == i
        plt.scatter(X_test[idx][:, 0], X_test[idx][:, 1], c=data_rgb.colors[i], s=30, marker=markers[i], alpha=0.75,
                    linestyle='dashed')
    # plt.ylabel("Feature #1")
    # plt.xlabel("Feature #0")
    plt.xticks([])
    plt.yticks([])
    plt.title(f"k-Nearest Neighbour (k={K})", font='sans serif')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.text(
        xx.max() - 0.1,
        yy.min() + 0.1,
        ("测试精度 : %.2f" % score).lstrip("0"),
        size=14,
        horizontalalignment="right",
    )
    plt.savefig(os.path.join('figures', f'ch16_two_moons_k={K}.pdf'), transparent=True, bbox_inches='tight')
    plt.close(fig)

Ks = 2 ** np.arange(int(np.ceil(np.log2(X_train.shape[0])))) + 1
Ks[0] -= 1
accs = []
for K in Ks:
    clf = KNeighborsClassifier(K)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accs.append(score)
fig = plt.figure()
plt.plot(accs, marker='o', color='red')
plt.xlabel("k")
plt.ylabel("kNN测试精度")
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
# plt.xscale('log')
plt.xticks(np.arange(len(Ks)), [str(k) for k in Ks])
plt.savefig(os.path.join('figures', f'ch16_k_acc_line.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)
