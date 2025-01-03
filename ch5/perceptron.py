import numpy as np


def train(data_x: np.ndarray, data_y: np.ndarray, w1: float, w2: float, theta: float, eta: float):
    train_x = np.concatenate([data_x, -1 * np.ones((4, 1))], axis=1)
    train_y = data_y
    weights = np.array([w1, w2, theta])

    def test():
        correct_num = 0
        for x, y in zip(train_x, train_y):
            activation = np.sum(weights * x)
            pred_y = int(activation >= -1e-3)
            correct_num += np.isclose(pred_y, y)
        return correct_num

    for ep in range(100):
        for x, y in zip(train_x, train_y):
            activation = np.sum(weights * x)
            # 考虑浮点数的计算误差，保证在取值为0时阶跃函数输出为1。
            pred_y = int(activation >= -1e-3)
            delta_w = eta * (y - pred_y) * x
            weights = weights + delta_w
        if test() == train_x.shape[0]:
            return w1, w2, theta
