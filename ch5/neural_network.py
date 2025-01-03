import numpy as np
import matplotlib.pyplot as plt


# 使用Sigmoid激活函数
def activation(x):
    return 1 / (1 + np.exp(-x))


# 需要学习的目标函数
def target_fn(x):
    return np.sin(np.pi * x)


# 前向过程实现
def forward(params, x):
    x1 = activation(x @ params["W0"] + params["b0"])
    x2 = activation(x1 @ params["W1"] + params["b1"])
    x3 = x2 @ params["W2"] + params["b2"]
    return {"X0": x, "X1": x1, "X2": x2, "X3": x3}


# 反向传播过程的实现
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


# 对预测结果进行评估，计算MSE
def evaluate(params, x, y):
    pred = forward(params, x)
    loss = np.mean((pred["X3"] - y) ** 2)
    return {"loss": loss, "pred_y": pred["X3"]}


# 进行一步梯度下降
def step(params, train_x, train_y, lr):
    pred = forward(params, train_x)
    grad = backward(params, pred, train_y)
    for k in params.keys():
        params[k] -= lr * grad[k]


def main():
    # 生成数据，我们对训练数据加一个小的扰动，以观察过拟合有噪声数据的影响
    train_x = np.random.uniform(-1, 1, size=(20, 1))
    train_y = target_fn(train_x) + np.random.randn(*train_x.shape) * 0.2
    test_x = np.linspace(-1, 1, 1000).reshape((1000, 1))
    test_y = target_fn(test_x)

    # 初始化网络参数
    params = {
        "W0": np.random.uniform(-1, 1, (1, 32)),
        "W1": np.random.uniform(-1, 1, (32, 32)),
        "W2": np.random.uniform(-1, 1, (32, 1)),
        "b0": np.ones((1, 32)) * 0.5,
        "b1": np.ones((1, 32)) * 0.5,
        "b2": np.ones((1, 1)) * 0.5
    }
    train_loss_hist = []
    test_loss_hist = []
    loss_step = []
    # 训练并画图
    for step_count in range(200000):
        # 一步梯度下降
        step(params, train_x, train_y, lr=1e-2)

        # 记录数据并画图
        if step_count % 500 == 0:
            # 计算训练集上的误差
            train_eval = evaluate(params, train_x, train_y)
            # 计算测试集上的误差
            test_eval = evaluate(params, test_x, test_y)
            # 记录误差
            train_loss_hist.append(train_eval["loss"])
            test_loss_hist.append(test_eval["loss"])
            loss_step.append(step_count)
            print("step_count {}, train_loss {}, test_loss {}".format(
                step_count, train_eval["loss"], test_eval["loss"]))
            # 每更新5000步画一次图
            if step_count % 5000 == 0:
                plt.scatter(train_x[:, 0], train_y[:, 0], label="train_data", c="blue")
                plt.plot(test_x[:, 0], test_eval["pred_y"][:, 0], label="predict", c="cyan")
                plot_x = np.linspace(-1, 1, 100).reshape((100,))
                plot_y = target_fn(plot_x)
                plt.plot(plot_x, plot_y, label="truth", c="red")
                plt.title("train mse {:.4f}, test mse {:.4f}".format(
                    train_eval["loss"], test_eval["loss"]))
                plt.legend()
                plt.savefig("./plots/step_{}_pred.jpg".format(step_count))
                plt.cla()
                plt.clf()
    plt.plot(loss_step, train_loss_hist, label="train")
    plt.plot(loss_step, test_loss_hist, label="test")
    plt.yscale("log")
    plt.legend()
    plt.savefig("./plots/loss.jpg")


if __name__ == "__main__":
    # 固定随机种子以保证可重复性
    np.random.seed(0)
    main()
