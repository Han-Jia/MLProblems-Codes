import numpy as np
from torch import permute


def plain_distance_function(X):
    D = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            D[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
    return D


def vectorized_distance_function(X):
    xTx = np.sum(X ** 2, axis=1).reshape((-1, 1))
    yTy = np.sum(X ** 2, axis=1).reshape((1, -1))
    xTy = X @ X.T
    square = xTx + yTy - 2 * xTy
    np.fill_diagonal(square, 0.)
    return np.sqrt(square)  # if +1e-13 before sqrt, then np.allclose will return False


def plain_permutation_function(X, p):
    permuted_X = np.zeros_like(X)
    for i in range(X.shape[0]):
        permuted_X[i] = X[p[i]]
    return permuted_X


def vectorized_permutation_function(X, p):
    row_idx = np.array(list(range(X.shape[0])))
    P = np.zeros((X.shape[0], X.shape[0]))
    P[row_idx, p] = 1
    permuted_X = P @ X
    return permuted_X


if __name__ == "__main__":
    import time

    m = 1000
    d = 20
    p = np.array(list(range(m)))
    X = np.random.randn(m, d)
    # np.random.shuffle(p)
    # x1 = plain_permutation_function(X, p)
    # x2 = vectorized_permutation_function(X, p)
    # print(np.mean(np.abs(x1 - x2)))
    # exit()
    t1 = time.time()
    d1 = plain_distance_function(X)
    t2 = time.time()
    d2 = vectorized_distance_function(X)
    t3 = time.time()
    print(t2 - t1)
    print(t3 - t2)
    print(np.mean(np.abs(d1 - d2)))
