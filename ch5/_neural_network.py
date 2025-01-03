import numpy as np
# import jax.numpy as jnp
# import jax.nn as jnn
# import jax
import matplotlib.pyplot as plt


def activation(x):
    return 1 / (1 + np.exp(-x))


def target_fn(x):
    return np.sin(np.pi * x)


# def forward(params, x):
#     x1 = activation(x @ params["W0"] + params["b0"])
#     x2 = x1 @ params["W1"] + params["b1"]
#     return {"X0": x, "X1": x1, "X2": x2}


# def jnp_loss(params, x, y):
#     x1 = jnn.sigmoid(jnp.add(jnp.matmul(x, params["W0"]), params["b0"]))
#     x2 = jnp.add(jnp.matmul(x1, params["W1"]), params["b1"])
#     loss = jnp.mean(jnp.power(x2 - y, 2) / 2)
#     return loss

# 
# def backward(params, pred, y):
#     grad = {"X2": (pred["X2"] - y) / np.prod(y.shape)}
#     grad["X1"] = grad["X2"] @ params["W1"].T
#     grad["W1"] = pred["X1"].T @ grad["X2"]
#     grad["W0"] = pred["X0"].T @ (grad["X1"] * pred["X1"] * (1 - pred["X1"]))
#     grad["b1"] = np.sum(grad["X2"], axis=0, keepdims=True)
#     # grad["b1"] = np.sum(grad["X2"] * pred["X2"] * (1 - pred["X2"]), axis=0, keepdims=True)
#     grad["b0"] = np.sum(grad["X1"] * pred["X1"] *
#                         (1 - pred["X1"]), axis=0, keepdims=True)
#     return grad


def forward(params, x):
    x1 = activation(x @ params["W0"] + params["b0"])
    x2 = activation(x1 @ params["W1"] + params["b1"])
    x3 = x2 @ params["W2"] + params["b2"]
    return {"X0": x, "X1": x1, "X2": x2, "X3": x3}


def backward(params, pred, y):
    grad = {"X3": (pred["X3"] - y) / np.prod(y.shape)}
    grad["X2"] = grad["X3"] @ params["W2"].T
    grad["X1"] = (grad["X2"] * pred["X2"] * (1 - pred["X2"])) @ params["W1"].T
    grad["W2"] = pred["X2"].T @ grad["X3"]
    grad["W1"] = pred["X1"].T @ (grad["X2"] * pred["X2"] * (1 - pred["X2"]))
    grad["W0"] = pred["X0"].T @ (grad["X1"] * pred["X1"] * (1 - pred["X1"]))
    grad["b2"] = np.sum(grad["X3"], axis=0, keepdims=True)
    grad["b1"] = np.sum(grad["X2"] * pred["X2"] * (1 - pred["X2"]), axis=0, keepdims=True)
    grad["b0"] = np.sum(grad["X1"] * pred["X1"] * (1 - pred["X1"]), axis=0, keepdims=True)
    return grad


def step(params, train_x, train_y, epoch):
    pred = forward(params, train_x)
    loss = np.mean((pred["X3"] - train_y) ** 2)
    # _loss = jnp_loss(params, train_x, train_y)
    # jax_grad_fn = jax.grad(jnp_loss, 0)
    # jax_grad = jax_grad_fn(params, train_x, train_y)
    # print(jax_grad)
    # exit()
    if epoch % 1000 == 0:
        print(loss)
        plt.scatter(train_x[:, 0], pred["X3"][:, 0])
        plt.scatter(train_x[:, 0], train_y[:, 0])
        plt.savefig("./tmp/tt.jpg")
        plt.cla()
        plt.clf()
        # print(loss, _loss)
    grad = backward(params, pred, train_y)
    for k in params.keys():
        # print(k)
        # print(grad[k])
        # print(jax_grad[k])
        # print(np.mean((grad[k] - jax_grad[k])**2))
        params[k] -= 1e-2 * grad[k]
    # exit()


def main():
    import os
    os.makedirs('./tmp', exist_ok=True)

    train_x = np.random.uniform(-1, 1, size=(100, 1))
    train_y = target_fn(train_x)
    test_x = np.random.uniform(-1, 1, size=(100, 1))
    test_y = target_fn(test_x)

    # params = {
    #     "W0": np.random.uniform(-1, 1, (1, 32)),
    #     "W1": np.random.uniform(-1, 1, (32, 1)),
    #     "b0": np.ones((1, 32)) * 0.5,
    #     "b1": np.ones((1, 1)) * 0.5,
    # }
    params = {
        "W0": np.random.uniform(-1, 1, (1, 32)),
        "W1": np.random.uniform(-1, 1, (32, 32)),
        "W2": np.random.uniform(-1, 1, (32, 1)),
        "b0": np.ones((1, 32)) * 0.5,
        "b1": np.ones((1, 32)) * 0.5,
        "b2": np.ones((1, 1)) * 0.5
    }
    for epoch in range(100000):
        step(params, train_x, train_y, epoch)


if __name__ == "__main__":
    np.random.seed(0)
    main()
