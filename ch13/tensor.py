import numpy as np


class Tensor:
    def __init__(self, value, grad_fn=lambda *args: None):
        self.np = value
        self.grad_fn = grad_fn
        self.shape = value.shape
        self.grad = np.zeros_like(value)

    def backward(self, grad_output=1.0):
        self.grad_fn(grad_output)
