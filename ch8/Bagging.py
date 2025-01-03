from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
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


def base_classification(X, y, algo):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    if algo == "DT":
        classifier = DecisionTreeClassifier(max_depth=5)
    elif algo == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=11)

    classifier.fit(train_x, train_y)
    pred_test_y = classifier.predict(test_x)

    acc = np.mean(pred_test_y == test_y)
    return acc


def random_select_classification(X, y, algo, ratio, T):
    # 划分训练测试数据
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # 特征维度, 随机选取的特征维度
    d, sd = X.shape[1], int(ratio * X.shape[1])

    pred_test_ys = []
    for _ in range(T):
        # 随机选择的特征的下标
        js = np.random.choice(np.arange(d), sd, replace=False)

        # 构造基分类器
        if algo == "DT":
            classifier = DecisionTreeClassifier(max_depth=10)
        elif algo == "KNN":
            classifier = KNeighborsClassifier(n_neighbors=11)

        # 在随机选定的训练集特征上训练
        classifier.fit(train_X[:, js], train_y)

        # 在随机选定的测试集特征上测试
        pred_test_y = classifier.predict(test_X[:, js])
        pred_test_ys.append(pred_test_y)

    # 投票进行集成并评估测试精度
    vote_test_y = [Counter(ys).most_common(1)[0][0] for ys in np.array(pred_test_ys).T]
    acc = np.mean(np.array(vote_test_y) == test_y)
    return acc


def random_extract_classification(X, y, algo, ratio, T):
    d = X.shape[1]

    # PCA提取特征
    pca = PCA(n_components=d)
    X = pca.fit_transform(X)

    acc = random_select_classification(X, y, algo, ratio, T)
    return acc


if __name__ == "__main__":
    X, y = datasets.load_digits(n_class=10, return_X_y=True)

    for algo in ["KNN", "DT"]:
        # 重复10次单个模型训练，当做基准
        base_accs = []
        for _ in range(10):
            base_acc = base_classification(X, y, algo)
            base_accs.append(base_acc)

        base_acc, base_acc_std = np.mean(base_accs), np.std(base_accs)

        ratios = np.linspace(0.1, 1.0, 10)
        print(ratios)

        # Random Subspace
        rs_accs = []
        re_accs = []

        for ratio in ratios:
            accs = []
            for _ in range(10):
                acc = random_select_classification(X, y, algo, ratio, _)
                accs.append(acc)

            rs_accs.append([np.mean(accs), np.std(accs)])

            accs = []
            for _ in range(10):
                acc = random_extract_classification(X, y, algo, ratio, _)
                accs.append(acc)

            re_accs.append([np.mean(accs), np.std(accs)])

        rs_accs = np.array(rs_accs)
        re_accs = np.array(re_accs)

        xs = list(range(len(ratios)))

        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        ax.plot(xs, rs_accs[:, 0], color="red", marker="o")
        ax.plot(xs, re_accs[:, 0], color="blue", marker="x")
        ax.hlines(base_acc, 0, len(xs) - 1, color="gray", linestyle="dashed")
        ax.legend(["随机选择特征", "随机提取特征", "单个模型基准"], fontsize=12)
        ax.set_xticks(xs)
        ax.set_xticklabels(["{:.1f}".format(ratio) for ratio in ratios])
        plt.setp(ax.get_xticklabels(), rotation=30)
        ax.set_xlabel("使用的特征比例", fontsize=16)
        ax.set_ylabel("精度", fontsize=16)
        ax.set_title(algo, fontsize=16)
        fig.tight_layout()
        plt.savefig(
            "ch8_bagging_{}.jpg".format(algo),
            dpi=300, bbox_inches=None
        )
        plt.show()
