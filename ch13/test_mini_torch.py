import re
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

import mini_torch as mtorch


def numerical_grad(function, inputs: List[np.ndarray], argnums=None, eps: float = 1e-6):
    if argnums is None:
        argnums = list(range(len(inputs)))
    grads = []
    for i in argnums:
        flat_x = inputs[i].reshape(-1)
        shape = inputs[i].shape
        grad = np.zeros_like(flat_x)
        for j in range(len(flat_x)):
            perturb_x = np.copy(flat_x)
            perturb_x[j] += eps
            inputs[i] = perturb_x.reshape(shape)
            f_plus = function(inputs)
            perturb_x[j] -= 2 * eps
            inputs[i] = perturb_x.reshape(shape)
            f_minus = function(inputs)
            perturb_x[j] += eps
            inputs[i] = perturb_x.reshape(shape)
            grad[j] = (f_plus - f_minus) / (2 * eps)
        grad = grad.reshape(shape)
        grads.append(grad)
    return grads


class MLP:

    def __init__(self, input_size, output_size):
        self.W1 = mtorch.Tensor(np.random.uniform(-1, 1,
                                                  size=(input_size, 32)), requires_grad=True)
        self.b1 = mtorch.Tensor(np.zeros((1, 32)), requires_grad=True)
        self.W2 = mtorch.Tensor(
            np.random.uniform(-1, 1, size=(32, 32)), requires_grad=True)
        self.b2 = mtorch.Tensor(np.zeros((1, 32)), requires_grad=True)
        self.W3 = mtorch.Tensor(
            np.random.uniform(-1, 1, size=(32, 32)), requires_grad=True)
        self.b3 = mtorch.Tensor(np.zeros((1, 32)), requires_grad=True)
        self.W4 = mtorch.Tensor(np.random.uniform(-1, 1,
                                                  size=(32, output_size)), requires_grad=True)
        self.b4 = mtorch.Tensor(np.zeros((1, output_size)), requires_grad=True)
        self.params = [self.W1, self.W2, self.W3, self.W4,
                       self.b1, self.b2, self.b3, self.b4]

    def forward(self, x):
        x1 = mtorch.add(mtorch.matmul(x, self.W1), self.b1)
        x1 = mtorch.relu(x1)
        x2 = mtorch.add(mtorch.matmul(x1, self.W2), self.b2)
        x2 = mtorch.relu(x2)
        x3 = mtorch.add(mtorch.matmul(x2, self.W3), self.b3)
        x3 = mtorch.add(mtorch.relu(x3), x1)
        x4 = mtorch.add(mtorch.matmul(x3, self.W4), self.b4)
        return x4
    
class MLPS:

    def __init__(self, input_size, output_size):
        self.W1 = mtorch.Tensor(np.random.uniform(-1, 1,
                                                  size=(input_size, 128)), requires_grad=True)
        self.b1 = mtorch.Tensor(np.zeros((1, 128)), requires_grad=True)
        self.W2 = mtorch.Tensor(
            np.random.uniform(-1, 1, size=(128, output_size)), requires_grad=True)
        self.b2 = mtorch.Tensor(np.zeros((1, output_size)), requires_grad=True)
        # self.W3 = mtorch.Tensor(
        #     np.random.uniform(-1, 1, size=(32, 32)), requires_grad=True)
        # self.b3 = mtorch.Tensor(np.zeros((1, 32)), requires_grad=True)
        # self.W4 = mtorch.Tensor(np.random.uniform(-1, 1,
        #                                           size=(32, output_size)), requires_grad=True)
        # self.b4 = mtorch.Tensor(np.zeros((1, output_size)), requires_grad=True)
        self.params = [self.W1, self.W2,
                       self.b1, self.b2]

    def forward(self, x):
        x1 = x @ self.W1 + self.b1
        x1 = mtorch.relu(x1)
        x2 = x1 @ self.W2 + self.b2
        return x2


def test_loss_function(inputs):
    np.random.seed(0)
    model = MLP(input_size=32, output_size=10)
    x = mtorch.Tensor(np.random.randn(100, 32))
    for i in range(len(model.params)):
        model.params[i].np = inputs[i]
    _y = model.forward()
    _y = mtorch.pow(_y, 2)
    loss = mtorch.mean(_y)
    return loss.np


def test_grad():
    np.random.seed(0)
    model = MLP(input_size=32, output_size=10)
    x = mtorch.Tensor(np.random.randn(100, 32))
    _y = model.forward(x)
    _y = mtorch.pow(_y, 2)
    loss = mtorch.mean(_y)
    loss.backward()
    grad = [p.grad for p in model.params]

    inputs = [t.np for t in model.params]
    num_grads = numerical_grad(test_loss_function, inputs)

    for i in range(len(grad)):
        dist = np.mean(np.abs(num_grads[i] - grad[i]))
        print(dist)


def onehot_encode(y, label_num):
    y_onehot = np.zeros((y.shape[0], label_num))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


def log_softmax(x):
    return mtorch.sub(x, mtorch.log(mtorch.sum(mtorch.exp(x), dim=1, keepdim=True)))


def test_digits():
    digits = datasets.load_digits()
    images = digits.images
    # images = images / np.max(images)
    targets = digits.target
    x = ((images - 8) / 16).reshape(-1, 64)

    x_train, x_test, y_train_t, y_test_t = train_test_split(
        x, targets, test_size=0.2, shuffle=True, stratify=targets
    )
    # _, axes = plt.subplots(nrows=10, ncols=10, figsize=(30, 30))
    # for ax, image, label in zip(axes.flat, images, targets):
    #     ax.set_axis_off()
    #     ax.imshow(image, cmap=plt.cm.gray_r)#, interpolation="nearest")
    #     ax.set_title("Training: %i" % label)
    # plt.savefig("./tmp/digits.png")

    y_train = onehot_encode(y_train_t, 10)
    # y_test = onehot_encode(y_test_t, 10)

    tensor_x_train = mtorch.Tensor(x_train)
    tensor_x_test = mtorch.Tensor(x_test)
    tensor_y_train = mtorch.Tensor(y_train)
    model = MLPS(input_size=64, output_size=10)
    optimizer = mtorch.SGDOptimizer(
        model.params, lr=1e-2, weight_decay=1e-5, momentum=0.9)

    for i in range(500000):
        # print("ss")
        logits = model.forward(tensor_x_train)
        # print(tensor_y.np.shape)
        # print(logits.np.shape)
        # exit()
        loss = mtorch.mean(mtorch.neg(
            mtorch.mul(tensor_y_train, log_softmax(logits))))
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            pred = np.argmax(logits.np, axis=1)
            acc = np.mean(pred == y_train_t)

            test_logits = model.forward(tensor_x_test)
            test_pred = np.argmax(test_logits.np, axis=1)
            test_acc = np.mean(test_pred == y_test_t)

            print("epoch {}, train loss {:.5f}, train_acc {:.5f}, test_acc {:.5f}".format(
                i, loss.np, acc, test_acc))


if __name__ == "__main__":
    # test_grad()
    test_digits()
