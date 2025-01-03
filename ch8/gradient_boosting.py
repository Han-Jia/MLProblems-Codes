from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from matplotlib import rcParams
from matplotlib import pyplot as plt
import numpy as np
import copy

plt.style.use("ggplot")

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def gradient_boosting_regressor(train_X, train_y, max_depth, T):
    res_train_y = copy.deepcopy(train_y)

    regressors = []
    weights = []
    for t in range(T):
        regressor = DecisionTreeRegressor(max_depth=max_depth)
        regressor.fit(train_X, res_train_y)
        pred_train_y = regressor.predict(train_X)

        weight = (res_train_y * pred_train_y).sum() / (pred_train_y ** 2).sum()

        regressors.append(regressor)
        weights.append(weight)

        res_train_y = res_train_y - pred_train_y
    return regressors, weights


if __name__ == "__main__":
    # 超参数
    T = 200
    for max_depth in [1, 5]:
        # 加载数据
        X, y = datasets.load_boston(return_X_y=True)
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 训练
        regressors, weights = gradient_boosting_regressor(
            train_x, train_y, max_depth, T
        )

        # 记录均方误差
        train_MSE = []
        test_MSE = []

        pred_train_ys = []
        pred_test_ys = []
        for t in range(T):
            pred_train_y = weights[t] * regressors[t].predict(train_x)
            pred_test_y = weights[t] * regressors[t].predict(test_x)
            pred_train_ys.append(pred_train_y)
            pred_test_ys.append(pred_test_y)

            train_MSE.append(
                ((np.sum(pred_train_ys, axis=0) - train_y) ** 2).mean()
            )
            test_MSE.append(
                ((np.sum(pred_test_ys, axis=0) - test_y) ** 2).mean()
            )

        # plot MSE
        xs = list(range(T))

        fig = plt.figure(figsize=(4, 3))
        plt.plot(xs, train_MSE, marker="+", markersize=4)
        plt.plot(xs, test_MSE, marker="x", markersize=4)
        plt.legend(["训练均方误差", "测试均方误差"], fontsize=14)
        plt.xlabel("基回归器数目", fontsize=16)
        plt.title("Max Depth={}".format(max_depth), fontsize=16)
        fig.tight_layout()
        fig.savefig(
            "ch8_boosting_{}.jpg".format(max_depth),
            dpi=300, bbox_inches=None
        )
        plt.show()
