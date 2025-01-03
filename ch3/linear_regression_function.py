import numpy as np


# 线性回归
def lin_reg(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    m = X_train.shape[0]
    one = np.full((m, 1), 1)
    coeff_x = (X_train.T @ X_train) - (1 / m) * (X_train.T @ one @ one.T @ X_train)
    coeff_y = (X_train.T) - (1 / m) * (X_train.T @ one @ one.T)
    return np.linalg.pinv(coeff_x) @ coeff_y @ y_train


# 岭回归
def ridge_reg(X_train: np.ndarray, y_train: np.ndarray, lmbd: float) -> np.ndarray:
    m = X_train.shape[0]
    d = X_train.shape[1]
    one = np.full((m, 1), 1)
    coeff_x = (X_train.T @ X_train) - (1 / m) * (X_train.T @ one @ one.T @ X_train) + lmbd * np.identity(d)
    coeff_y = (X_train.T) - (1 / m) * (X_train.T @ one @ one.T)
    return np.linalg.pinv(coeff_x) @ coeff_y @ y_train


# 计算MSE, 根据lmbd是否输入判断是否岭回归
def MSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, lmbd: float = None) -> float:
    m = X_test.shape[0]
    one = np.full((m,), 1)
    if lmbd is not None:
        W = ridge_reg(X_train, y_train, lmbd)  # 调用岭回归函数求解W
    else:
        W = lin_reg(X_train, y_train)  # 调用基础线性回归函数求解W
    b = np.sum(y_train - (X_train @ W)) / len(y_train)  # 使用同样的方法求解偏移b
    pred = X_test @ W + one * b  # 使用W和b进行预测
    return np.sum(np.square(pred - y_test)) / y_test.shape[0]
