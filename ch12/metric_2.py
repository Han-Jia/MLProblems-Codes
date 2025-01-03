from imblearn.datasets import fetch_datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
import numpy as np

# 读取数据集
dataset = fetch_datasets()["yeast_me2"]
x, y = dataset.data, dataset.target
# 划分训练集和测试集
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(x, y)

acc_list, mean_acc_list = [], []
for train_index, test_index in skf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 训练svm模型
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    # 评估模型在测试集上的准确率与平均准确率
    result = clf.predict(x_test)
    acc = (result == y_test).mean()
    acc_pos = ((result == y_test) & (y_test == -1)).sum() / (y_test == -1).sum()
    acc_neg = ((result == y_test) & (y_test == 1)).sum() / (y_test == 1).sum()
    mean_acc = (acc_pos + acc_neg) / 2
    acc_list.append(acc)
    mean_acc_list.append(mean_acc)
# 计算五折交叉验证后的平均结果
acc = sum(acc_list) / len(acc_list)
mean_acc = sum(mean_acc_list) / len(mean_acc_list)
print(acc, mean_acc)
