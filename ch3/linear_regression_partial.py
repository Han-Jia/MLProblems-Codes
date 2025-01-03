# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_boston(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
# print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


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


report_lin_reg_MSE = lambda: lin_reg_MSE(train_x, train_y, test_x, test_y)
report_lin_reg_MSE_selected = lambda: lin_reg_MSE(train_x[:50, :], train_y[:50], test_x, test_y)


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


report_ridge_reg_MSE = lambda lmbd: ridge_reg_MSE(train_x, train_y, test_x, test_y, lmbd)
report_ridge_reg_MSE_selected = lambda lmbd: ridge_reg_MSE(train_x[:50, :], train_y[:50], test_x, test_y, lmbd)

print("MSE is", report_lin_reg_MSE())
print("MSE is", report_ridge_reg_MSE(1))


# below is the implement of "Absorb b into w"

def add_margin(X: np.ndarray) -> np.ndarray:
    X_1margin = np.full((X.shape[0], 1), 1)
    return np.concatenate((X, X_1margin), axis=1)


def lin_reg_absorb(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    X_hat = add_margin(X_train)
    return (np.linalg.pinv((X_hat.T) @ X_hat) @ (X_hat.T)) @ y_train


def lin_reg_MSE_absorb(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    W = lin_reg_absorb(X_train, y_train)
    pred = add_margin(X_test) @ W
    return np.sum(np.square(pred - y_test)) / y_test.shape[0]


report_lin_reg_MSE_absorb = lambda: lin_reg_MSE_absorb(train_x, train_y, test_x, test_y)
report_lin_reg_MSE_absorb_selected = lambda lmbd: ridge_reg_MSE(train_x[:50, :], train_y[:50], test_x, test_y, lmbd)


def ridge_reg_absorb(X_train: np.ndarray, y_train: np.ndarray, lmbd: float) -> np.ndarray:
    X_hat = add_margin(X_train)
    return (np.linalg.pinv(((X_hat.T) @ X_hat) + lmbd * np.identity(X_hat.shape[1])) @ (X_hat.T)) @ y_train


def ridge_reg_MSE_absorb(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                         lmbd: float) -> float:
    W = ridge_reg_absorb(X_train, y_train, lmbd)
    pred = add_margin(X_test) @ W
    return np.sum(np.square(pred - y_test)) / y_test.shape[0]


report_ridge_reg_MSE_absorb = lambda lmbd: ridge_reg_MSE_absorb(train_x, train_y, test_x, test_y, lmbd)
report_ridge_reg_MSE_absorb_selected = lambda lmbd: ridge_reg_MSE_absorb(train_x[:50, :], train_y[:50], test_x, test_y,
                                                                         lmbd)

report_lin_reg_MSE_absorb()
# print( "MSE is", report_lin_reg_MSE_absorb() )
# print( "MSE is", report_ridge_reg_MSE_absorb(1) )

grid_x = np.array([])
grid_x = np.concatenate((grid_x, np.linspace(0, 5, 50)))
# grid_x = np.concatenate((grid_x, np.logspace(0, np.log10(5), 64)))
grid_x = grid_x.tolist()
grid_x.sort()

import matplotlib
from matplotlib import pyplot as plt

config = {
    'font.family': 'serif',
    'font.serif': ['simHei'],
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
    'font.size': 18,
}
plt.rcParams.update(config)

# matplotlib.rcParams['figure.figsize'] = [8, 6]
# matplotlib.rc('font', **font)
# matplotlib.rc('axes', linewidth=2)
# matplotlib.rcParams['font.sans-serif'] = ['SimSun']
# matplotlib.rcParams['axes.unicode_minus']=False
# plt.title("MSE-$\lambda$")
plt.xlabel("$\lambda$", size=22)
plt.ylabel("MSE", size=22)
plt.scatter(grid_x, [report_lin_reg_MSE() for i in grid_x], label=u"线性回归")
plt.scatter(grid_x, [report_ridge_reg_MSE(i) for i in grid_x], marker='d', label=u"岭回归")
# plt.scatter(grid_x,[reportRidgeRegMSEAbsorb(i) for i in grid_x],label="Ridge Regression (Absorb $b$ into $w$)")
plt.legend()
# plt.show()
plt.savefig('FullDataMSE.pdf')

# matplotlib.rcParams['figure.figsize'] = [8, 6]
# matplotlib.rc('font', **font)
# matplotlib.rc('axes', linewidth=2)
# matplotlib.rcParams['font.sans-serif'] = ['SimSun']
# matplotlib.rcParams['axes.unicode_minus']=False
# plt.title("MSE-$\lambda$ based on Selected Training Set")
plt.clf()
plt.xlabel("$\lambda$", size=22)
plt.ylabel("MSE", size=22)
plt.scatter(grid_x, [report_lin_reg_MSE_selected() for i in grid_x], label=u"线性回归")
plt.scatter(grid_x, [report_ridge_reg_MSE_selected(i) for i in grid_x], marker='d', label=u"岭回归")
# plt.scatter(grid_x,[reportRidgeRegMSEAbsorb_selected(i) for i in grid_x],label="Ridge Regression (Absorb $b$ into $w$)")
plt.legend()
# plt.show()
plt.savefig('PartialDataMSE.pdf')
