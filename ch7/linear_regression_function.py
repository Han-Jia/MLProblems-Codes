import numpy as np


# linear regression
def lin_reg(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    m = X_train.shape[0]
    one = np.full((m, 1), 1)
    coeff_x = (X_train.T @ X_train) - (1 / m) * (X_train.T @ one @ one.T @ X_train)
    coeff_y = (X_train.T) - (1 / m) * (X_train.T @ one @ one.T)
    return np.linalg.pinv(coeff_x) @ coeff_y @ y_train


def lin_reg_MSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    m = X_test.shape[0]
    one = np.full((m,), 1)
    W = lin_reg(X_train, y_train)
    b = np.sum(y_train - (X_train @ W)) / len(y_train)
    pred = X_test @ W + one * b
    return np.sum(np.square(pred - y_test)) / y_test.shape[0]


# ridge regression
def ridge_reg(X_train: np.ndarray, y_train: np.ndarray, lmbd: float) -> np.ndarray:
    m = X_train.shape[0]
    d = X_train.shape[1]
    one = np.full((m, 1), 1)
    coeff_x = (X_train.T @ X_train) - (1 / m) * (X_train.T @ one @ one.T @ X_train) + lmbd * np.identity(d)
    coeff_y = (X_train.T) - (1 / m) * (X_train.T @ one @ one.T)
    return np.linalg.pinv(coeff_x) @ coeff_y @ y_train


def ridge_reg_MSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                  lmbd: float) -> float:
    m = X_test.shape[0]
    one = np.full((m,), 1)
    W = ridge_reg(X_train, y_train, lmbd)
    b = np.sum(y_train - (X_train @ W)) / len(y_train)
    pred = X_test @ W + one * b
    return np.sum(np.square(pred - y_test)) / y_test.shape[0]
