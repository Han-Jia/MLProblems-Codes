# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
from sklearn.metrics import f1_score
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据集
covtype = fetch_covtype()
X, y = covtype.data, covtype.target
print(X.shape)
_, y = np.unique(y, return_inverse=True)
n_classes = np.unique(y).shape[0]
inst_per_cls = np.array([np.where(y == idx)[0].shape[0] for idx in range(n_classes)])
# 先构造平衡数据集
X, y = make_imbalance(X, y,
                      sampling_strategy=dict(zip(range(n_classes), [min(inst_per_cls)] * n_classes)),
                      random_state=2022)
# 划分训练集和测试集，注意使用分层划分，保证类别比例不变
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2022, stratify=y
)
# 对样本特征进行归一化，这有利于模型快速收敛
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 再构造不平衡数据集
inst_per_cls = np.array([np.where(y_train == idx)[0].shape[0] for idx in range(n_classes)])
inst_num_per_cls = [np.min([int(np.ceil(np.max(inst_per_cls) * (0.01 ** (idx / (n_classes - 1.0))))),
                            inst_per_cls[idx]]) for idx in range(n_classes)]
x_train_imbalance, y_train_imbalance = make_imbalance(X_train, y_train,
                                                      sampling_strategy=dict(zip(range(n_classes), inst_num_per_cls)),
                                                      random_state=2022)
inst_per_cls = np.array([np.where(y_test == idx)[0].shape[0] for idx in range(n_classes)])
inst_num_per_cls = [np.min([int(np.ceil(np.max(inst_per_cls) * (0.01 ** (idx / (n_classes - 1.0))))),
                            inst_per_cls[idx]]) for idx in range(n_classes)]
x_test_imbalance, y_test_imbalance = make_imbalance(X_test, y_test,
                                                    sampling_strategy=dict(zip(range(n_classes), inst_num_per_cls)),
                                                    random_state=2022)

# 分别在平衡训练集和不平衡训练集上训练svm模型
clf_balance = LR(multi_class='multinomial', max_iter=1000)
clf_balance.fit(X_train, y_train)
clf_imbalance = LR(multi_class='multinomial', max_iter=1000)
clf_imbalance.fit(x_train_imbalance, y_train_imbalance)

# balance+balance
predict_bb = clf_balance.predict(X_test)
macro_f1_bb = f1_score(y_test, predict_bb, average='macro')
micro_f1_bb = f1_score(y_test, predict_bb, average='micro')
print('macro_f1_bb:{}'.format(macro_f1_bb))
print('micro_f1_bb:{}'.format(micro_f1_bb))

# balance+imbalance
predict_bib = clf_balance.predict(x_test_imbalance)
macro_f1_bib = f1_score(y_test_imbalance, predict_bib, average='macro')
micro_f1_bib = f1_score(y_test_imbalance, predict_bib, average='micro')
print('macro_f1_bib:{}'.format(macro_f1_bib))
print('micro_f1_bib:{}'.format(micro_f1_bib))

# imbalance+balance
predict_ibb = clf_imbalance.predict(X_test)
macro_f1_ibb = f1_score(y_test, predict_ibb, average='macro')
micro_f1_ibb = f1_score(y_test, predict_ibb, average='micro')
print('macro_f1_ibb:{}'.format(macro_f1_ibb))
print('micro_f1_ibb:{}'.format(micro_f1_ibb))

# imbalance+imbalance
predict_ibib = clf_imbalance.predict(x_test_imbalance)
macro_f1_ibib = f1_score(y_test_imbalance, predict_ibib, average='macro')
micro_f1_ibib = f1_score(y_test_imbalance, predict_ibib, average='micro')
print('macro_f1_ibib:{}'.format(macro_f1_ibib))
print('micro_f1_ibib:{}'.format(micro_f1_ibib))
