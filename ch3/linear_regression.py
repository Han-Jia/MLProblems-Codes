from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_boston(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)


# 线性回归
def lin_reg(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    pass


# 岭回归
def ridge_reg(X_train: np.ndarray, y_train: np.ndarray, lmbd: float) -> np.ndarray:
    pass


# 计算MSE, 根据lmbd是否输入判断是否岭回归
def MSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, lmbd: float = None) -> float:
    pass


# 针对基本线性回归和岭回归模型计算MSE
report_lin_reg_MSE = lambda: MSE(train_x, train_y, test_x, test_y)
report_ridge_reg_MSE = lambda lmbd: MSE(train_x, train_y, test_x, test_y, lmbd)
