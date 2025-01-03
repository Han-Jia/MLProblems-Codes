from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
# 出于效率考虑, 取前10个feature
X = X[:, :10]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)


def transform_tuple(tuples, y):
    '''
    :参数 tuples: 样本对, 形状为(n,2,d), n为样本对数量, d为样本的维度
    :参数 y: 样本对的标记, 形状为(n,), 1表示正样本对, 0表示负样本对
    :返回 tuple_X : 转化后的样本对特征, 形状为(n, d^2)
    :返回 tuple_y : 转化后的样本对标记, 形状为(n,)
    '''
    return tuple_X, tuple_y


def transform_triplet(triplets):
    '''
    :参数 triplets: 三元组, 形状为(n,3,d), n为样本对数量, d为样本的维度
    :返回 triplet_X : 转化后的三元组特征
    :返回 triplet_y : 转化后的三元组标记

    线性分类器至少需要两个类的样本，思考如何构建转化后的三元组特征与标记使其适用于线性分类器
    '''
    return triplet_X, triplet_y


# 将现有的样本对转化为线性模型的特征以及标记(此处省略了tuples的构建过程)
tuple_X, tuple_y = transform_tuple(tuples, y)
# 创建二元组二分类器
tuple_lr = LogisticRegression(max_iter=1000)
tuple_lr.fit(tuple_X, tuple_y)
# 将现有的三元组转化为线性模型的特征以及标记(此处省略了triplets的构建过程)
triplet_X, triplet_y = transform_triplet(triplets)
# 创建三元组二分类器
triplets_lr = LogisticRegression(max_iter=1000)
# 为了正常训练一个二分类器, 将(a,p,n)作为负类, (a,n,p)作为正类
triplets_lr.fit(triplet_X, triplet_y)
# 二元组分类器的"马氏矩阵", 以线性分类器的权重向量表示
M = tuple_lr.coef_


def m_dist(x1: np.ndarray, x2: np.ndarray, M: np.ndarray):
    '''
    该函数给定样本对和"马氏矩阵"得到样本对之间的"距离"
    '''
    diff = np.expand_dims(x1 - x2, axis=1)
    return np.sum((diff * diff.T).reshape(-1) * M)


# 训练一个以线性分类器权重为"度量"的k近邻分类器
tuple_knn = KNeighborsClassifier(metric=partial(m_dist, M=M))
tuple_knn.fit(X_train, y_train)
# 评估分类器的性能
acc_tuple = tuple_knn.score(X_test, y_test)
