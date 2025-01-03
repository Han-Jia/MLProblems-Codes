from sklearn.datasets import make_blobs, make_moons
import numpy as np
import matplotlib.pyplot as plt

# 构造凸的数据集
X_convex, _ = make_blobs(
    n_samples=[200, 200],
    centers=[[-1.0, -1.0], [1.0, 1.0]],
    cluster_std=[0.3, 0.3],
    random_state=1,
)
# 构造非凸的数据集
X_nonconvex, _ = make_moons(
    n_samples=[200, 200],
    random_state=1,
)
# 构造非凸且含有噪声的数据集
x3 = np.linspace(0.0, 1.0, 20)
y3 = np.linspace(0.15, 0.25, 3)
YY, XX = np.meshgrid(y3, x3)
X_new = np.c_[XX.ravel(), YY.ravel()]
X_noise = np.concatenate((X_new, X_nonconvex), 0)
