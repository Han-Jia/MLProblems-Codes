import numpy as np

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

np.random.seed(0)


# 绘制决策边界
def plot_decision_boundary(model, x):
    # 生成网格点坐标矩阵,得到两个矩阵
    M, N = 500, 500
    x0, x1 = np.meshgrid(np.linspace(x[:, 0].min(), x[:, 0].max(), M), np.linspace(x[:, 1].min(), x[:, 1].max(), N))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    z = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#FFC0CB', '#B0E0E6'])
    plt.pcolormesh(x0, x1, z, cmap=custom_cmap)


def get_dataset(name):
    n_samples = 1700
    if name == 'noisy_circles':
        noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
        X, y = noisy_circles
        return X, y

    if name == 'blob':
        blobs = datasets.make_blobs(n_samples=n_samples, random_state=170, centers=2)
        X, y = blobs
        transformation = [[0.4, -0.5], [-0.3, 0.8]]
        X_aniso = np.dot(X, transformation)
        X = X_aniso
        return X, y


if __name__ == "__main__":
    X, y = get_dataset('blob')

    # add your code for the task
