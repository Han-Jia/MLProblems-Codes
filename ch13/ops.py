import numpy as np
from matmul import FunctionWrapper


class Tensor:
    # 针对Tensor的加法运算
    def __add__(self, other):
        return add(self, other)


# 作为运算操作的基类
class Function:
    def __init__(self):
        self.save_for_backward = None

    def backward(self, grad_output):
        pass


# 用于将前向过程中被NumPy通过广播（broadcasting）方式自动扩展维度的Tensor的梯度进行维度还原，梯度为g，原始向量大小为shape
def _unbroadcast_grad(g, shape):
    # 梯度维度和原始Tensor维度一致，满足要求，直接返回
    if g.shape == shape:
        return g
    assert len(g.shape) == len(shape)
    # 依次查看各维度，判断哪些维度被进行广播
    dims = tuple([i for i in range(len(g.shape))
                  if shape[i] < g.shape[i]])
    # 对被通过广播所扩展的维度进行求和
    return np.sum(g, axis=dims, keepdims=True)


# 实现Tensor的加法操作
class AddFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        # 存储输入Tensor A和B用于反向传播
        self.save_for_backward = [A, B]
        # 基于NumPy得到加法运算结果
        res = A.np + B.np
        # 基于NumPy的结果res构造Tensor类，并设置该Tensor梯度计算函数为当前加法函数的backward函数
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        # 提取前向运算中的Tensor
        A, B = self.save_for_backward
        # 对梯度grad_output的大小进行调整，使其和A，B大小一致
        A_grad = _unbroadcast_grad(grad_output, A.np.shape)
        B_grad = _unbroadcast_grad(grad_output, B.np.shape)
        assert A_grad.shape == A.np.shape and B_grad.shape == B.np.shape
        if A.requires_grad:
            A.grad += A_grad  # 计算加法结果对A的梯度
        if B.requires_grad:
            B.grad += B_grad  # 计算加法结果对B的梯度
        A.grad_fn(A_grad)  # 递归调用Tensor A对应的梯度计算函数
        B.grad_fn(B_grad)  # 递归调用Tensor B对应的梯度计算函数
        return A_grad, B_grad


# 实现ReLU算子操作
class ReLUFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.save_for_backward = X
        # 基于NumPy得到ReLU激活函数的结果
        res = np.clip(X.np, a_min=0, a_max=None)
        # 基于NumPy的结果res构造Tensor类，并设置该Tensor梯度计算函数为当前加法函数的backward函数
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        # 提取前向运算中的Tensor
        X = self.save_for_backward
        # 查看原始Tensor X中哪些元素非负
        mask = X.np >= 0
        input_grad = grad_output * mask
        X.grad += input_grad  # 计算ReLU算子结果对X的梯度
        assert input_grad.shape == X.np.shape
        X.grad_fn(input_grad)  # 递归调用Tensor X对应的梯度计算函数
        return input_grad


add = FunctionWrapper(AddFunction).apply
relu = FunctionWrapper(ReLUFunction).apply
