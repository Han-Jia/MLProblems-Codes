import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 使用欧几里得距离作为度量
clf = KNeighborsClassifier(K, metric='euclidean')
# 在训练数据上计算协方差矩阵
covariace = np.cov(X_train.T)
# 使用马氏距离作为度量，使用马氏距离时，需要通过metric_params传入所需参数，即协方差矩阵
clf = KNeighborsClassifier(K, metric='mahalanobis', metric_params={'V': covariace})
