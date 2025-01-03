import time

import numpy as np
from metric_learn import LMNN
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
import os
from sklearn.linear_model import LogisticRegression

import matplotlib
from matplotlib import pyplot as plt

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


def transform_feature(x1: np.ndarray, x2: np.ndarray):
    diff = np.expand_dims(x1 - x2, axis=1)
    return (diff * diff.T).reshape(-1)


def m_dist(x1: np.ndarray, x2: np.ndarray, M: np.ndarray):
    diff = np.expand_dims(x1 - x2, axis=1)
    return np.sum((diff * diff.T).reshape(-1) * M)


def create_tuple_triplet(X, cls_idxes, num_pos=-1, num_neg=-1, random_pos=False, random_neg=False):
    distances = euclidean_distances(X)
    for i in range(X.shape[0]):
        distances[i, i] = 1e8
    triplets = []
    tuples, y = [], []
    for cls, cls_idx in enumerate(cls_idxes):
        all_idxes = np.arange(X.shape[0])
        bool_pos_idx = np.zeros(X.shape[0], dtype=bool)
        bool_pos_idx[cls_idx] = True
        ncls_idx = all_idxes[~bool_pos_idx]
        for i, a in enumerate(cls_idx):
            dists = distances[a]
            pos_dists = dists[bool_pos_idx]
            neg_dists = dists[~bool_pos_idx]
            sorted_pos_idx = np.argsort(pos_dists)
            sorted_neg_idx = np.argsort(neg_dists)
            if random_pos:
                pos_idx = np.random.choice(np.delete(cls_idx, i), num_pos)
            else:
                pos_idx = cls_idx[sorted_pos_idx[:num_pos]]
            if random_neg:
                neg_idx = np.random.choice(ncls_idx, num_neg)
            else:
                neg_idx = ncls_idx[sorted_neg_idx[:num_neg]]
            for p in pos_idx:
                tuples.append(transform_feature(X[a], X[p]))
                y.append(0)
                for n in neg_idx:
                    triplets.append(transform_feature(X[a], X[p]) - transform_feature(X[a], X[n]))
            for n in neg_idx:
                tuples.append(transform_feature(X[a], X[n]))
                y.append(1)
    return np.array(triplets), np.array(tuples), np.array(y)


def hist(X, y, M, desc='tuple_class'):
    y = np.expand_dims(y, axis=1)
    distances = pairwise_distances(X, metric=partial(m_dist, M=M))
    cls_indicator_matrix = y == y.T
    self_mask = (np.eye(X.shape[0]) - np.ones_like(distances)).astype(bool)
    same_cls_dist = distances[cls_indicator_matrix & self_mask]
    diff_cls_dist = distances[(~cls_indicator_matrix) & self_mask]
    fig = plt.figure()
    plt.hist(same_cls_dist, bins=100, density=True, label='Same', alpha=0.5)
    plt.hist(diff_cls_dist, bins=100, density=True, label='Different', alpha=0.5)
    plt.legend(prop={"family": "Times New Roman"})
    plt.xticks(font='sans serif')
    plt.yticks(font='sans serif')
    plt.xlim([-10, 200])
    plt.ylim([0, 0.04])
    plt.savefig(os.path.join('figures', f'ch16_{desc}_hist.pdf'), transparent=True, bbox_inches='tight')
    plt.close(fig)


os.makedirs('figures', exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)
X = X[:, :10]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)
unique_y = np.unique(y_train)
cls_idxes = []
arange = np.arange(X_train.shape[0])
for l in unique_y:
    cls_idx = arange[y_train == l]
    cls_idxes.append(cls_idx)
params = [
    {'num_pos': -1, 'num_neg': -1, 'random_pos': False, 'random_neg': False},
    {'num_pos': 10, 'num_neg': -1, 'random_pos': True, 'random_neg': False},
    {'num_pos': 10, 'num_neg': -1, 'random_pos': False, 'random_neg': False},
    {'num_pos': 10, 'num_neg': 15, 'random_pos': True, 'random_neg': True},
    {'num_pos': 10, 'num_neg': 15, 'random_pos': True, 'random_neg': False},
    {'num_pos': 10, 'num_neg': 15, 'random_pos': False, 'random_neg': True},
    {'num_pos': 10, 'num_neg': 15, 'random_pos': False, 'random_neg': False},
]

knn_ = KNeighborsClassifier()
knn_.fit(X_train, y_train)
acc_raw = knn_.score(X_test, y_test)
print(acc_raw)
triplets, tuples, tuple_y = create_tuple_triplet(X_train, cls_idxes, **params[-1])

tuple_clf = LogisticRegression(max_iter=1000)
tuple_clf.fit(tuples, tuple_y)
M = tuple_clf.coef_

tuple_knn = KNeighborsClassifier(metric=partial(m_dist, M=M))
# tuple_knn = KNeighborsClassifier(metric=m_dist)
tuple_knn.fit(X_train, y_train)
acc_tuple = tuple_knn.score(X_test, y_test)
print(acc_tuple)
hist(X, y, M, desc='tuple_class')

triplets_clf = LogisticRegression(max_iter=1000)
num_triplets = len(triplets)
triplets_clf.fit(np.concatenate([triplets, -triplets]), np.array([0] * num_triplets + [1] * num_triplets))
M = triplets_clf.coef_

triplet_knn = KNeighborsClassifier(metric=partial(m_dist, M=M))
triplet_knn.fit(X_train, y_train)
acc_triplet = triplet_knn.score(X_test, y_test)
print(acc_triplet)

hist(X, y, M, desc='triplet_class')

lmnn = LMNN()
lmnn.fit(X_train, y_train)
M = lmnn.get_mahalanobis_matrix().reshape(-1)

lmnn_knn = KNeighborsClassifier(metric=partial(m_dist, M=M))
lmnn_knn.fit(X_train, y_train)
acc_lmnn = lmnn_knn.score(X_test, y_test)
print(acc_lmnn)

hist(X, y, M, desc='lmnn_class')

fig = plt.figure()
accs = [acc_raw, acc_lmnn, acc_tuple, acc_triplet]
plt.bar(['raw', 'LMNN', 'tuple LR', 'triplet LR'], accs)
plt.ylim([np.min(accs) - 0.05, np.max(accs) + 0.01])
plt.ylabel('kNN测试精度')
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
plt.savefig(os.path.join('figures', f'ch16_class_method_acc.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)
