from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)


def linear_regression(X_train, y_train):
    ''' 线性回归
    :参数 X_train: np.ndarray, 形状为(n, d), n个d维训练样本
    :参数 y_train: np.ndarray, 形状为(n, 1), 每个样本的标签
    :返回: 权重矩阵W
    '''
    pass


def ridge_regression(X_train, y_train, lmbd):
    ''' 岭回归
    :参数 X_train: np.ndarray, 形状为(n, d), n个d维训练样本
    :参数 y_train: np.ndarray, 形状为(n, 1), 每个样本的标签
    :参数 lmbd: float, 岭回归lambda参数
    :返回: 权重矩阵W
    '''
    pass


def MSE(X_train, y_train, X_test, y_test, lmbd=None):
    ''' 计算MSE, 根据lmbd是否输入判断是否岭回归
    :参数 X_train: np.ndarray, 形状为(n, d), n个d维训练样本
    :参数 y_train: np.ndarray, 形状为(n, 1), 每个训练样本的标签
    :参数 X_test: np.ndarray, 形状为(m, d), m个d维测试样本
    :参数 y_test: np.ndarray, 形状为(m, 1), 每个测试样本的标签
    :参数 lmbd: float或None, 岭回归&\codecommentcolor{}$\lambda$&参数, None表示使用线性回归
    :返回: 标量, MSE值
    '''
    pass


# 针对基本线性回归和岭回归模型计算MSE
linear_regression_MSE = lambda: MSE(train_x, train_y, test_x, test_y)
ridge_regression_MSE = lambda lmbd: MSE(train_x, train_y, test_x, test_y, lmbd)
