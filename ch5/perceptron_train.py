import numpy as np


def train(X, y, eta, epoch):
    # 增加哑节点
    X = np.concatenate(
        [X, -1 * np.ones((X.shape[0], 1))], axis=1)
    # 初始化权重向量
    w = np.random.uniform(-1, 1, X.shape[1])
    for ep in range(epoch):
        update_count = 0
        for x, y_ in zip(X, y):
            activation = w.T @ x
            # 考虑浮点数的计算误差，保证在取值为0时阶跃函数输出为1。
            pred_y = int(activation >= -1e-3)
            # 根据感知机的更新公式计算更新量
            delta_w = eta * (y_ - pred_y) * x
            # 更新权重
            w = w + delta_w
            if y_ != pred_y:
                update_count += 1
        # 如果在一轮训练中都没有进行更新, 则感知机已经收敛
        if update_count == 0:
            return w
