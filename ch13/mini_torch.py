import numpy as np


class Tensor:

    def __init__(self, value, requires_grad=True, grad_fn=lambda *args: None):
        self.np = value
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.shape = value.shape
        if requires_grad:
            self.grad = np.zeros_like(value)
        else:
            self.grad = None

    def backward(self, grad_output=1.0):
        self.grad_fn(grad_output)

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __neg__(self):
        return neg(self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __exp__(self, other):
        return exp(self, other)

    def __repr__(self) -> str:
        return str(self.np)


class Function:

    def __init__(self):
        self.save_for_backward = None

    def backward(self, grad_output):
        pass


class MeanFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X, dim=None, keepdim=False):
        res = np.mean(X.np, axis=dim, keepdims=keepdim)
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        self.save_for_backward = [X, dim, keepdim]
        return res_t

    def backward(self, grad_output):
        X, dim, keepdim = self.save_for_backward
        if not X.requires_grad:
            return
        if dim is None:
            grad_input = np.broadcast_to(
                grad_output / np.prod(X.np.shape), X.np.shape)
        else:
            if keepdim:
                grad_input = np.broadcast_to(
                    grad_output / X.np.shape[dim], X.np.shape)
            else:
                grad_input = np.broadcast_to(np.expand_dims(
                    grad_output, axis=dim) / X.np.shape[dim], X.np.shape)
        assert grad_input.shape == X.np.shape
        X.grad += grad_input
        X.grad_fn(grad_input)
        return grad_input


class SumFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X, dim=None, keepdim=False):
        res = np.sum(X.np, axis=dim, keepdims=keepdim)
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        self.save_for_backward = [X, dim, keepdim]
        return res_t

    def backward(self, grad_output):
        X, dim, keepdim = self.save_for_backward
        if not X.requires_grad:
            return
        if dim is None:
            grad_input = np.broadcast_to(
                grad_output, X.np.shape)
        else:
            if keepdim:
                grad_input = np.broadcast_to(
                    grad_output, X.np.shape)
            else:
                grad_input = np.broadcast_to(np.expand_dims(
                    grad_output, axis=dim), X.np.shape)
        assert grad_input.shape == X.np.shape
        X.grad += grad_input
        X.grad_fn(grad_input)
        return grad_input


def _unbroadcast_grad(g, shape):
    if g.shape == shape:
        return g
    assert len(g.shape) == len(shape)
    dims = tuple([i for i in range(len(g.shape))
                  if shape[i] < g.shape[i]])
    return np.sum(g, axis=dims, keepdims=True)


class AddFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        self.save_for_backward = [A, B]
        res = A.np + B.np
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        A, B = self.save_for_backward
        Agrad = _unbroadcast_grad(grad_output, A.np.shape)
        Bgrad = _unbroadcast_grad(grad_output, B.np.shape)
        assert Agrad.shape == A.np.shape and Bgrad.shape == B.np.shape
        if A.requires_grad:
            A.grad += Agrad
        if B.requires_grad:
            B.grad += Bgrad
        A.grad_fn(Agrad)
        B.grad_fn(Bgrad)
        return Agrad, Bgrad


class MulFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        self.save_for_backward = [A, B]
        res = A.np * B.np
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        A, B = self.save_for_backward
        Agrad = _unbroadcast_grad(grad_output * B.np, A.np.shape)
        Bgrad = _unbroadcast_grad(grad_output * A.np, B.np.shape)
        assert Agrad.shape == A.np.shape and Bgrad.shape == B.np.shape
        if A.requires_grad:
            A.grad += Agrad
        if B.requires_grad:
            B.grad += Bgrad
        A.grad_fn(Agrad)
        B.grad_fn(Bgrad)
        return Agrad, Bgrad


class SubFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        self.save_for_backward = [A, B]
        res = A.np - B.np
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        A, B = self.save_for_backward
        Agrad = _unbroadcast_grad(grad_output, A.np.shape)
        Bgrad = - _unbroadcast_grad(grad_output, B.np.shape)
        assert Agrad.shape == A.np.shape and Bgrad.shape == B.np.shape
        if A.requires_grad:
            A.grad += Agrad
        if B.requires_grad:
            B.grad += Bgrad
        A.grad_fn(Agrad)
        B.grad_fn(Bgrad)
        return Agrad, Bgrad


class PowerFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X, p):
        self.save_for_backward = [X, p]
        res = np.power(X.np, p)
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        X, p = self.save_for_backward
        input_grad = p * np.power(X.np, p - 1) * grad_output
        assert input_grad.shape == X.np.shape
        if X.requires_grad:
            X.grad += input_grad
        X.grad_fn(input_grad)
        return input_grad


class MatMulFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        self.save_for_backward = [A, B]
        res = np.matmul(A.np, B.np)
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        A, B = self.save_for_backward
        if A.requires_grad:
            Agrad = np.matmul(grad_output, B.np.T)
            assert Agrad.shape == A.np.shape
            A.grad += Agrad
            A.grad_fn(Agrad)
        if B.requires_grad:
            Bgrad = np.matmul(A.np.T, grad_output)
            assert Bgrad.shape == B.np.shape
            B.grad += Bgrad
            B.grad_fn(Bgrad)
        return Agrad, Bgrad


class ReLUFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.save_for_backward = X
        res = np.clip(X.np, a_min=0, a_max=None)
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        X = self.save_for_backward
        mask = X.np >= 0
        input_grad = grad_output * mask
        X.grad += input_grad
        assert input_grad.shape == X.np.shape
        X.grad_fn(input_grad)
        return input_grad


class ExpFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.save_for_backward = X
        res = np.exp(X.np)
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        X = self.save_for_backward
        input_grad = grad_output * np.exp(X.np)
        X.grad += input_grad
        assert input_grad.shape == X.np.shape
        X.grad_fn(input_grad)
        return input_grad


class LogFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.save_for_backward = X
        res = np.log(X.np)
        # print(res)
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        X = self.save_for_backward
        input_grad = 1.0 / (X.np + 1e-6) * grad_output
        X.grad += input_grad
        assert input_grad.shape == X.np.shape
        X.grad_fn(input_grad)
        return input_grad


class NegFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.save_for_backward = X
        res = -X.np
        res_t = Tensor(res, requires_grad=True, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        X = self.save_for_backward
        input_grad = -1 * grad_output
        X.grad += input_grad
        assert (np.isscalar(input_grad) and isinstance(
            X.np, float)) or input_grad.shape == X.np.shape
        X.grad_fn(input_grad)
        return input_grad

class SGDOptimizer:

    def __init__(self, params, lr, weight_decay=0, momentum=0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.step_count = 0
        self.mu = [np.zeros_like(p) for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.np)

    def step(self):
        self.step_count += 1
        for i, p in enumerate(self.params):
            if self.step_count == 1:
                self.mu[i] = p.grad
            else:
                self.mu[i] = self.momentum * self.mu[i] + \
                    (1 - self.momentum) * p.grad
        for p, m in zip(self.params, self.mu):
            p.np = (1 - self.lr * self.weight_decay) * p.np - m * self.lr


class FunctionWrapper:

    def __init__(self, Fn):
        self.Fn = Fn

    def apply(self, *args, **kwargs):
        fn = self.Fn()
        return fn.forward(*args, **kwargs)


mean = FunctionWrapper(MeanFunction).apply
matmul = FunctionWrapper(MatMulFunction).apply
pow = FunctionWrapper(PowerFunction).apply
add = FunctionWrapper(AddFunction).apply
sub = FunctionWrapper(SubFunction).apply
relu = FunctionWrapper(ReLUFunction).apply
exp = FunctionWrapper(ExpFunction).apply
log = FunctionWrapper(LogFunction).apply
sum = FunctionWrapper(SumFunction).apply
mul = FunctionWrapper(MulFunction).apply
neg = FunctionWrapper(NegFunction).apply
