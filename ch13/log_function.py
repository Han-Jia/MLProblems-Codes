import numpy as np
from tensor import Tensor


class LogFunction:
    def forward(self, X):
        self.save_for_backward = X
        res = np.log(X.np)
        res_t = Tensor(res, grad_fn=self.backward)
        return res_t

    def backward(self, grad_output):
        X = self.save_for_backward
        input_grad = 1.0 / (X.np + 1e-6) * grad_output
        X.grad += input_grad
        X.grad_fn(input_grad)
        return input_grad
