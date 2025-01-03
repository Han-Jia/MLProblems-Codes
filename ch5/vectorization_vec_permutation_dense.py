import numpy as np


def vectorized_permutation_dense_function(X, p):
    row_idx = np.array(list(range(X.shape[0])))
    P = np.zeros((X.shape[0], X.shape[0]))
    P[row_idx, p] = 1
    permuted_X = P @ X
    return permuted_X
