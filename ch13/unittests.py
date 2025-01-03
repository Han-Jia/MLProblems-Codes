from turtle import pos
import numpy as np
import mini_torch as mtorch
from test_mini_torch import numerical_grad

def random_tensor(shape, positive=False):
    if positive:
        return mtorch.Tensor(np.random.uniform(0.5, 10, size=shape))
    else:
        return mtorch.Tensor(np.random.uniform(-1, 1, size=shape))

class RandomTensor:
    def __init__(self, shapes, positive=False):
        self.shapes = shapes
        self.positive = positive
    def __call__(self):
        return [random_tensor(shape, self.positive) for shape in self.shapes]

class Wrapper:
    
    def __init__(self, fn):
        self.fn = fn
        
    def __call__(self, inputs):
        if isinstance(inputs[0], np.ndarray):
            _inputs = [mtorch.Tensor(x) for x in inputs]
            return self.fn(_inputs).np
        else:
            return self.fn(inputs)
        
def _numerical_grad(function, inputs, argnums=None, eps: float = 1e-6):
    if isinstance(inputs[0], mtorch.Tensor):
        _inputs = [x.np for x in inputs]
        return numerical_grad(function, _inputs, argnums, eps)
    else:
        return numerical_grad(function, inputs, argnums, eps)
    
def _check_grad(inputs_list, fn_list):
    for rt_fn in inputs_list:
        for Fn in fn_list:
            x = rt_fn()
            y = Fn(x)
            y.backward()
            grad = _numerical_grad(Fn, x)
            for i in range(len(x)):
                print("check diff {}".format(np.mean(np.abs(grad[i] - x[i].grad))))
                assert np.allclose(grad[i], x[i].grad)

def test_mean_function():
    inputs_list = [
        RandomTensor([(100, 32)]),
        RandomTensor([(100, 1)]),
        RandomTensor([(1, 32)]),
    ]
    def fn1(inputs):
        return mtorch.mean(inputs[0])
    Fn1 = Wrapper(fn1)
    
    def fn2(inputs):
        return mtorch.mean(mtorch.mean(inputs[0], dim=0))
    Fn2 = Wrapper(fn2)
    
    def fn3(inputs):
        return mtorch.mean(mtorch.mean(inputs[0], dim=1))
    Fn3 = Wrapper(fn3)
    
    fn_list = [Fn1, Fn2, Fn3]
    _check_grad(inputs_list, fn_list)
    
def test_log_function():
    inputs_list = [
        RandomTensor([(100, 32)], positive=True),
        RandomTensor([(100, 1)], positive=True),
        RandomTensor([(1, 32)], positive=True),
    ]
    def fn1(inputs):
        return mtorch.mean(mtorch.log(inputs[0]))
    Fn1 = Wrapper(fn1)
    
    fn_list = [Fn1]
    _check_grad(inputs_list, fn_list)
    
def test_exp_function():
    inputs_list = [
        RandomTensor([(100, 32)]),
        RandomTensor([(100, 1)]),
        RandomTensor([(1, 32)]),
    ]
    def fn1(inputs):
        return mtorch.mean(mtorch.exp(inputs[0]))
    Fn1 = Wrapper(fn1)
    
    fn_list = [Fn1]
    _check_grad(inputs_list, fn_list)
    
    
def test_neg_function():
    inputs_list = [
        RandomTensor([(100, 32)]),
        RandomTensor([(100, 1)]),
        RandomTensor([(1, 32)]),
    ]
    def fn1(inputs):
        return mtorch.mean(-inputs[0])
    Fn1 = Wrapper(fn1)
    
    fn_list = [Fn1]
    _check_grad(inputs_list, fn_list)
    
                
def test_add_function():
    inputs_list = [
        RandomTensor([(100, 32), (100, 32)]),
        RandomTensor([(100, 1), (100, 32)]),
        RandomTensor([(100, 32), (1, 32)]),
        RandomTensor([(1, 1), (100, 32)]),
    ]
    def fn1(inputs):
        return mtorch.mean(inputs[0] + inputs[1])
    Fn1 = Wrapper(fn1)
    fn_list = [Fn1]
    
    _check_grad(inputs_list, fn_list)
    
    
def test_matmul_function():
    inputs_list = [
        RandomTensor([(100, 32), (32, 4)]),
        RandomTensor([(1, 100), (100, 32)]),
        RandomTensor([(1, 100), (100, 1)]),
        RandomTensor([(200, 1), (1, 200)]),
    ]
    def fn1(inputs):
        return mtorch.mean(inputs[0] @ inputs[1])
    Fn1 = Wrapper(fn1)
    fn_list = [Fn1]
    
    _check_grad(inputs_list, fn_list)
    