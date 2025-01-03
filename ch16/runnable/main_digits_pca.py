# @author : Administrator 
# @date : 2022/4/22
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

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

# 读取手写数字数据集
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=0
)

os.makedirs('figures', exist_ok=True)

accuracy = []
ns = np.arange(1, X.shape[1])
for n in ns:
    pca = make_pipeline(StandardScaler(), PCA(n_components=n, random_state=0))
    knn = KNeighborsClassifier(n_neighbors=3)
    # Fit the method's model
    pca.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(pca.transform(X_train), y_train)

    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(pca.transform(X_test), y_test)

    accuracy.append(acc_knn)
    # Embed the data set in 2 dimensions using the fitted model
    # X_embedded = pca.transform(X)

fig = plt.figure()
plt.plot(ns, accuracy, marker='o', color='red')
plt.xlabel("n")
plt.ylabel("kNN测试精度")
plt.xticks(font='sans serif')
plt.yticks(font='sans serif')
# plt.xscale('log')
# plt.xticks(np.arange(len(Ks)),[str(k) for k in Ks])
plt.savefig(os.path.join('figures', f'ch16_n_acc_line.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)
