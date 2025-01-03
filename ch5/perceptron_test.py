import numpy as np


def test(X, y, w):
    # 测试感知机的准确性
    # 增加哑节点
    X = np.concatenate(
        [X, -1 * np.ones((X.shape[0], 1))], axis=1)
    correct_num = 0
    for x, y_ in zip(X, y):
        activation = w.T @ x
        # 考虑浮点数的计算误差，保证在取值为0时阶跃函数输出为1。
        pred_y = int(activation >= -1e-3)
        correct_num += np.isclose(pred_y, y_)
    return correct_num
