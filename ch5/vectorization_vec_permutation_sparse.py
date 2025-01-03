import numpy as np
from scipy.sparse import csc_matrix


def vectorized_permutation_sparse_function(X, p):
    row_idx = np.array(list(range(X.shape[0])))
    # 使用稀疏矩阵实现重排列, 注意csc_matrix的参数为(值, (行索引, 列索引)), 矩阵形状
    P = csc_matrix((np.ones(X.shape[0]), (row_idx, p)), shape=(X.shape[0], X.shape[0]))
    # 矩阵乘法实现重排列
    permuted_X = P @ X
    return permuted_X
