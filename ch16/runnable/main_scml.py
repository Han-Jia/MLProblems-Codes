import time

import numpy as np
from metric_learn import SCML
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import os

import random

import matplotlib
from matplotlib import pyplot as plt

config = {
    'font.family': 'serif',
    'font.serif': ['kaiti'],
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
    'font.size': 12,
}
plt.rcParams.update(config)

matplotlib.rc('axes', linewidth=2)

seed = 3
random.seed(seed)
np.random.seed(seed)


def parse_str(param):
    if param['num_pos'] == -1:
        pos_str = 'all'
    else:
        if param['random_pos']:
            stra = 'rd'
        else:
            stra = 'nn'
        pos_str = f'{param["num_pos"]} {stra}'
    if param['num_neg'] == -1:
        neg_str = 'all'
    else:
        if param['random_neg']:
            stra = 'rd'
        else:
            stra = 'nn'
        neg_str = f'{param["num_neg"]} {stra}'
    return f'{pos_str}\n{neg_str}'


def create_triplet(X, cls_idxes, num_pos=-1, num_neg=-1, random_pos=False, random_neg=False):
    distances = euclidean_distances(X)
    for i in range(X.shape[0]):
        distances[i, i] = np.inf
    triplets = []
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
                pos_idx = np.random.choice(np.delete(cls_idx, i), num_pos, replace=False)
            else:
                pos_idx = cls_idx[sorted_pos_idx[:num_pos]]
            if random_neg:
                neg_idx = np.random.choice(ncls_idx, num_neg, replace=False)
            else:
                neg_idx = ncls_idx[sorted_neg_idx[:num_neg]]
            for p in pos_idx:
                for n in neg_idx:
                    triplets.append([a, p, n])
    return triplets


def evaluate_triplet(X_train, cls_idxes, triplet_params, y_train, X_test, y_test):
    triplets = create_triplet(X_train, cls_idxes, **triplet_params)
    scml = SCML(preprocessor=X_train, random_state=0)
    start = time.time()
    scml.fit(triplets)
    span = time.time() - start
    X_tr = scml.transform(X_train)
    X_t = scml.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_tr, y_train)
    acc = knn.score(X_t, y_test)
    print(f'param : {triplet_params} num_samples: {len(triplets)} accuracy : {acc}')
    return len(triplets), span, acc


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.25)
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

os.makedirs('figures', exist_ok=True)

spans, accs = [], []
for param in params:
    num_samples, span, acc = evaluate_triplet(X_train, cls_idxes, param, y_train, X_test, y_test)
    spans.append(span)
    accs.append(acc)
fig = plt.figure()
plt.bar([parse_str(param) for param in params], accs)
plt.axhline(acc_raw, linestyle='--', color='k')
plt.ylim([np.min(accs) - 0.05, np.max(accs) + 0.01])
plt.ylabel('kNN测试精度')
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
plt.savefig(os.path.join('figures', f'ch16_triplets_accs.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.bar([parse_str(param) for param in params], spans)
# plt.ylim([np.min(spans) - 0.05, np.max(spans) + 0.01])
plt.ylabel('时间(秒)')
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
plt.savefig(os.path.join('figures', f'ch16_triplets_time.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)
