import numpy as np


# Tensor类中的其他部分（如构造函数）请参考问题13.4
class Tensor:
    # 针对Tensor的矩阵乘法运算，其他部分省略
    def __matmul__(self, other):
        return matmul(self, other)


from tensor import Tensor
# 实现矩阵乘法操作
class MatMulFunction:
    # 矩阵乘法的前向运算
    def forward(self, A, B):
        # 存储输入矩阵A和B用于反向传播
        self.save_for_backward = [A, B]
        # 基于NumPy得到加法运算结果
        res = np.matmul(A.np, B.np)
        # 基于NumPy的结果res构造Tensor类，并设置该Tensor梯度计算函数为当前矩阵乘法函数的backward函数
        res_t = Tensor(res, grad_fn=self.backward)
        return res_t

    # 矩阵乘法的反向运算
    def backward(self, grad_output):
        # 提取前向运算中的Tensor，对应矩阵A和B
        A, B = self.save_for_backward
        # 矩阵乘法out = A * B结果对A、B分别计算梯度
        if A.requires_grad:
            # 计算矩阵乘法结果对A的梯度
            A_grad = np.matmul(grad_output, B.np.T)
            assert A_grad.shape == A.np.shape
            A.grad += A_grad
            # 递归调用Tensor A对应的梯度计算函数
            A.grad_fn(A_grad)
        if B.requires_grad:
            B_grad = np.matmul(A.np.T, grad_output)
            assert B_grad.shape == B.np.shape
            B.grad += B_grad
            B.grad_fn(B_grad)
        return A_grad, B_grad


# 将运算类进行封装，在运算过程中自动调用其forward函数
class FunctionWrapper:
    def __init__(self, Fn):
        self.Fn = Fn

    def apply(self, *args, **kwargs):
        fn = self.Fn()
        return fn.forward(*args, **kwargs)


matmul = FunctionWrapper(MatMulFunction).apply
