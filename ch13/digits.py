import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mini_torch as mtorch
from neural_net import MLP


def log_softmax(x):
    return mtorch.sub(x, mtorch.log(mtorch.sum(mtorch.exp(x), dim=1, keepdim=True)))


def onehot_encode(y, label_num):
    y_onehot = np.zeros((y.shape[0], label_num))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


digits = datasets.load_digits()
images = digits.images
targets = digits.target
x = ((images - 8) / 16).reshape(-1, 64)

x_train, x_test, y_train_t, y_test_t = train_test_split(
    x, targets, test_size=0.2, shuffle=True, stratify=targets)

y_train = onehot_encode(y_train_t, 10)

tensor_x_train = mtorch.Tensor(x_train)
tensor_x_test = mtorch.Tensor(x_test)
tensor_y_train = mtorch.Tensor(y_train)
model = MLP(input_size=64, output_size=10)
optimizer = mtorch.SGDOptimizer(model.params, lr=1e-2, weight_decay=1e-5, momentum=0.9)

for i in range(500000):
    logits = model.forward(tensor_x_train)
    loss = mtorch.mean(mtorch.neg(
        mtorch.mul(tensor_y_train, log_softmax(logits))))
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
