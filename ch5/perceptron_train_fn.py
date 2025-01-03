import numpy as np

X = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])  # 训练数据, 输入都相同为X
y_and = np.array([0, 0, 0, 1])  # 与函数对应的y
y_or = np.array([0, 1, 1, 1])  # 或函数对应的y
y_xor = np.array([0, 1, 1, 0])  # 异或函数对应的y


def train(X, y, eta, epoch):
    ''' 实现训练感知机的算法, 并返回训练好的权重向量w
    :参数 X: 训练数据
    :参数 y: 数据标签
    :参数 eta: 感知机的参数&\codecommentcolor{}$\eta$&
    :参数 epoch: 训练轮数
    :返回: 感知机的权重w
    '''
    pass
