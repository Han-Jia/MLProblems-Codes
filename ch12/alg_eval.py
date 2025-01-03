import random
import numpy as np

from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class SMOTE(object):
    def __init__(self, X, y, N, K, random_state=0):
        self.N = N  # 每个少数类样本合成样本个数
        self.K = K  # k近邻个数
        self.label = y  # 进行数据增强的类别
        self.sample = X
        self.n_sample, self.n = self.sample.shape  # 获得样本个数，特征个数

    def over_sampling(self):
        if self.N < 100:
            idx = random.sample(range(self.n_sample), int(self.N * self.n_sample))
            return self.sample[idx], np.ones(len(idx)) * self.label
        else:
            N = int(self.N / 100) - 1
            synthetic = []  # 生成样本
            neighbors = NearestNeighbors(n_neighbors=self.K + 1).fit(self.sample)  # 计算每个样本的k近邻
            for i in range(self.n_sample):
                synthetic_sample = self.popluate(neighbors, i, N)  # 生成样本
                synthetic.append(synthetic_sample)
            synthetic_sample = np.vstack(synthetic)  # 将生成样本合并
            return np.vstack((self.sample, synthetic_sample)), (np.ones((N + 1) * self.n_sample) * self.label).astype(
                int)

    def popluate(self, neighbors, i, N):
        nnarray = neighbors.kneighbors(self.sample[i].reshape(1, -1), return_distance=False)[0][1:]  # 获得i样本的k近邻
        syn_sample = []
        for j in range(N):
            nn = np.random.randint(0, self.K)  # 随机选择一个k近邻
            nn = nnarray[nn]  # 获得k近邻索引
            dif = self.sample[nn] - self.sample[i]  # 获得k近邻与i样本的差值
            gap = random.random()
            syn_sample.append(self.sample[i] + gap * dif)  # 生成一个新样本
        return np.vstack(syn_sample)


class EasyEnsemble():
    def __init__(self, n_estimators=10, base_classifier="LR"):
        self.n_estimators = n_estimators  # 分类器个数
        self.base_classifier = base_classifier  # 基分类器

    def get_base_classifier(self):
        if self.base_classifier.lower() == "lr":
            return LogisticRegression(max_iter=3000)
        if self.base_classifier == "decisiontree":
            return DecisionTreeClassifier(max_depth=20, min_samples_split=20, min_samples_leaf=5)

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.estimators = []
        pos_num = np.sum(y == 1)
        neg_num = len(y) - pos_num
        if pos_num > neg_num:
            major_class = 1
            self.major_size = pos_num
            self.minor_size = neg_num
        else:
            major_class = 0
            self.major_size = neg_num
            self.minor_size = pos_num
        minor_x, major_x = X[y == 1 - major_class], X[y == major_class]  # 分离多数类和少数类
        estimators = []  # 分类器
        for i in range(self.n_estimators):
            balance_major_x = major_x[np.random.choice(self.major_size, self.minor_size, replace=False)]  # 多数类样本随机欠采样
            cls = AdaBoostClassifier(self.get_base_classifier(),
                                     algorithm="SAMME",
                                     n_estimators=200, learning_rate=0.2)  # 定义Adaboost，使用LogisticRegression作为基分类器
            under_sample_x = np.vstack((balance_major_x, minor_x))  # 少数类样本和欠采样的多数类样本合并
            under_sample_y = np.hstack(
                (np.ones(self.minor_size) * major_class, np.ones(self.minor_size) * (1 - major_class))).astype(
                int)  # 少数类样本和欠采样的多数类样本标签合并
            cls.fit(under_sample_x, under_sample_y)  # 训练
            estimators.append(cls)  # 存储分类器
        self.estimators = estimators

    def predict(self, X):
        classes = self.classes[:, np.newaxis]
        y_pred = []
        y_weight = []
        for e in self.estimators:
            pred = sum((child_estimator.predict(X) == classes).T * w
                       for child_estimator, w in zip(e.estimators_, e.estimator_weights_)
                       )  # 获取Adaboost中每个分类器的预测
            sum_weight = e.estimator_weights_.sum()
            y_pred.append(pred)
            y_weight.append(sum_weight)
        y_pred = sum(y_pred) / sum(y_weight)  # 计算Adaboost基分类器的集成的预测
        y_pred = (y_pred[:, 1] > 0.5).astype(int)  # 计算预测结果
        return y_pred


# 计算模型性能
def model_eval(y_list, pred_list):
    print("Accuracy:{:.1f}".format(
        100 * sum([balanced_accuracy_score(y_list[i], pred_list[i]) for i in range(len(y_list))]) / len(y_list)))
    print("F1-score(Class 1):{:.1f}".format(
        100 * sum([f1_score(y_list[i], pred_list[i]) for i in range(len(y_list))]) / len(y_list)))
    print("F1-score(Class 0):{:.1f}".format(
        100 * sum([f1_score(1 - y_list[i], 1 - pred_list[i]) for i in range(len(y_list))]) / len(y_list)))
    print("\n")


if __name__ == '__main__':
    # 读取, 预处理数据
    sat_image = fetch_datasets()["yeast_me2"]
    X, y = sat_image.data, sat_image.target
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y = ((y + 1) / 2).astype(int)

    # 不加任何处理
    pred_list = []
    y_list = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob[:, 1] > 0.5).astype(int)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("Logistic Regression performance:")
    model_eval(y_list, pred_list)

    # 使用阈值移动
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        pos_num = sum(y_train == 1)
        neg_num = sum(y_train == 0)
        threshold = 1 - 1 / (1 + (pos_num / neg_num))
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob[:, 1] > threshold).astype(int)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("LR + Threshold-moving performance:")
    model_eval(y_list, pred_list)

    # 加入类别权重
    pred_list = []
    y_list = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(class_weight="balanced", max_iter=3000)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob[:, 1] > 0.5).astype(int)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("Reweight(LR) performance:")
    model_eval(y_list, pred_list)

    # 使用SMOTE
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        pos_num = sum(y_train == 1)
        neg_num = sum(y_train == 0)
        smote = SMOTE(X=X_train[y_train == 1], y=1, N=neg_num / pos_num * 100, K=5)
        smote_X, smote_y = smote.over_sampling()
        sampled_X = np.vstack((smote_X, X_train[y_train == 0]))
        sampled_y = np.hstack((smote_y, np.zeros(neg_num))).astype(int)
        clf = LogisticRegression(max_iter=3000)
        clf.fit(sampled_X, sampled_y)
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob[:, 1] > 0.5).astype(int)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("SMOTE(LR) performance:")
    model_eval(y_list, pred_list)

    # 加入EasyEnsemble
    pred_list = []
    y_list = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = EasyEnsemble()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("EasyEnsemble(LR) performance:")
    model_eval(y_list, pred_list)

    # 使用决策树作为分类器
    pred_list = []
    y_list = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = DecisionTreeClassifier(max_depth=20, min_samples_split=20, min_samples_leaf=5)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob[:, 1] > 0.5).astype(int)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("DecisionTree performance:")
    model_eval(y_list, pred_list)

    # 使用决策树作为分类器, 并使用Reweight
    pred_list = []
    y_list = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = DecisionTreeClassifier(max_depth=20, min_samples_split=20, min_samples_leaf=5)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob[:, 1] > 0.5).astype(int)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("Reweight(DT) performance:")
    model_eval(y_list, pred_list)

    # 使用决策树作为分类器，并使用SMOTE
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        pos_num = sum(y_train == 1)
        neg_num = sum(y_train == 0)
        smote = SMOTE(X=X_train[y_train == 1], y=1, N=neg_num / pos_num * 100, K=5)
        smote_X, smote_y = smote.over_sampling()
        sampled_X = np.vstack((smote_X, X_train[y_train == 0]))
        sampled_y = np.hstack((smote_y, np.zeros(neg_num))).astype(int)
        clf = DecisionTreeClassifier(max_depth=20, min_samples_split=20, min_samples_leaf=5)
        clf.fit(sampled_X, sampled_y)
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob[:, 1] > 0.5).astype(int)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("DT + SMOTE performance:")
    model_eval(y_list, pred_list)

    # 使用决策树作为基分类器的Easy Ensemble
    pred_list = []
    y_list = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = EasyEnsemble(base_classifier="DecisionTree")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pred_list.append(y_pred)
        y_list.append(y_test)
    print("EasyEnsemble(DT) performance:")
    model_eval(y_list, pred_list)
