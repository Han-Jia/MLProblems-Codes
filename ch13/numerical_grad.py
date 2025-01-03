import numpy as np


def numerical_grad(function, inputs, argnums=None, eps=1e-6):
    # 如果没有指定argnums，那么就认为是求所有参数的梯度
    if argnums is None:
        argnums = list(range(len(inputs)))
    grads = []
    for i in argnums:
        # 为了便于操作, 首先把输入变成一维
        flat_x = inputs[i].reshape(-1)
        shape = inputs[i].shape
        # 初始化梯度值
        grad = np.zeros_like(flat_x)
        # 对每个元素进行扰动并记录梯度值
        for j in range(len(flat_x)):
            perturb_x = np.copy(flat_x)
            # 正向扰动
            perturb_x[j] += eps
            inputs[i] = perturb_x.reshape(shape)
            # 计算正向扰动的函数值
            f_plus = function(inputs)
            # 负向扰动
            perturb_x[j] -= 2 * eps
            inputs[i] = perturb_x.reshape(shape)
            # 计算负向扰动的函数值
            f_minus = function(inputs)
            perturb_x[j] += eps
            inputs[i] = perturb_x.reshape(shape)
            # 计算数值梯度
            grad[j] = (f_plus - f_minus) / (2 * eps)
        # 将梯度矩阵转化为原来的形状
        grad = grad.reshape(shape)
        grads.append(grad)
    return grads
