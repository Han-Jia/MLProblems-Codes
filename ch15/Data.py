import numpy as np

from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split


def load_data(dataset, domain_shift=False):
    if dataset == "BREASTCANCER":
        X, Y = datasets.load_breast_cancer(return_X_y=True)
    elif dataset == "DIGITS":
        X, Y = datasets.load_digits(n_class=10, return_X_y=True)
    elif dataset == "USPS":
        X, Y = fetch_openml(data_id=41082, return_X_y=True)
        Y = np.array([int(y) - 1 for y in Y])
    else:
        raise ValueError("No such dataset: {}".format(dataset))

    # 归一化
    X = MinMaxScaler().fit_transform(X)

    # 划分数据：训练集、测试集1、测试集2
    n_classes = len(np.unique(Y))

    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, test_size=0.5, random_state=42
    )
    te_X1, te_X2, te_Y1, te_Y2 = train_test_split(
        te_X, te_Y, test_size=0.5, random_state=42
    )

    # 分布偏移场景加噪声
    if domain_shift is True:
        tr_X += 0.1 * np.random.rand(*tr_X.shape)

        n_sam = int(0.05 * tr_X.shape[0])
        for _ in range(n_sam):
            k = np.random.randint(0, len(tr_Y))
            tr_Y[k] = np.random.randint(0, n_classes)

        te_X2 += 0.05 * np.random.randn(*te_X2.shape)

    return tr_X, tr_Y, te_X1, te_Y1, te_X2, te_Y2
