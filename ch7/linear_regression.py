from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_boston(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)


# linear regression
def lin_reg(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    pass


def lin_reg_MSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    pass


report_lin_reg_MSE = lambda: lin_reg_MSE(train_x, train_y, test_x, test_y)


# ridge regression
def ridge_reg(X_train: np.ndarray, y_train: np.ndarray, lmbd: float) -> np.ndarray:
    pass


def ridge_reg_MSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                  lmbd: float) -> float:
    pass


report_ridge_reg_MSE = lambda lmbd: ridge_reg_MSE(train_x, train_y, test_x, test_y, lmbd)
