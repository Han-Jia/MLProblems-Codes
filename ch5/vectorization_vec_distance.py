import numpy as np


def vectorized_distance_function(X):
    # x^Tx
    xTx = np.sum(X ** 2, axis=1).reshape((-1, 1))
    # y^Ty
    yTy = np.sum(X ** 2, axis=1).reshape((1, -1))
    # 2x^Ty
    xTy = X @ X.T
    square = xTx + yTy - 2 * xTy
    np.fill_diagonal(square, 0.)
    return np.sqrt(square)  # if +1e-13 before sqrt, then np.allclose will return False
