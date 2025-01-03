from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter

from matplotlib import rcParams
from matplotlib import pyplot as plt

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


def base_decision_tree(train_x, train_y, test_x):
    max_depth = np.random.choice(range(1, 20))
    min_samples_split = np.random.choice(range(2, 10))

    model = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_split=min_samples_split
    )
    model.fit(train_x, train_y)

    predy = model.predict(test_x)
    return predy


def decision_tree_ensemble(train_x, train_y, test_x, test_y, T):
    pred_ys = []
    judge_arr = []
    for _ in range(T):
        pred_y = base_decision_tree(train_x, train_y, test_x)
        pred_ys.append(pred_y)
        judge_arr.append((pred_y == test_y).astype(np.int32))
    pred_ys = np.array(pred_ys).T
    vote_pred_y = [Counter(ys).most_common(1)[0][0] for ys in pred_ys]
    return np.array(judge_arr), np.array(vote_pred_y)


def calculate_disagreement_measure(judge_arr):
    a = np.dot(judge_arr, judge_arr.T).sum()
    b = np.dot(judge_arr, (1 - judge_arr).T).sum()
    c = np.dot(1 - judge_arr, judge_arr.T).sum()
    d = np.dot(1 - judge_arr, (1 - judge_arr).T).sum()
    return (b + c) / (a + b + c + d)


def plot_correlation(n_group, dataset):
    # 加载数据
    if dataset == "iris":
        X, y = datasets.load_iris(return_X_y=True)
    elif dataset == "wine":
        X, y = datasets.load_wine(return_X_y=True)
    elif dataset == "digits":
        X, y = datasets.load_digits(n_class=10, return_X_y=True)
    else:
        raise ValueError("Dataset {} not supported.".format(dataset))

    dis_ms = []
    ens_accs = []

    for _ in range(n_group):
        T = np.random.choice(range(20, 100))
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
        judge_arr, vote_pred_y = decision_tree_ensemble(
            train_x, train_y, test_x, test_y, T
        )
        dis_m = calculate_disagreement_measure(judge_arr)
        dis_ms.append(dis_m)
        ens_accs.append(np.mean(vote_pred_y == test_y))

    fig = plt.figure(figsize=(3, 3))
    plt.scatter(dis_ms, ens_accs, marker="x", color="#666666")
    plt.xlabel("不合度量", fontsize=16)
    plt.ylabel("集成分类器精度", fontsize=14)
    plt.title(dataset.upper(), fontsize=16)
    fig.tight_layout()
    fig.savefig(
        "ch8_correlation-{}.jpg".format(dataset),
        dpi=300, bbox_inches=None
    )
    plt.show()


if __name__ == "__main__":
    plot_correlation(n_group=100, dataset="iris")
    plot_correlation(n_group=100, dataset="wine")
    plot_correlation(n_group=100, dataset="digits")
