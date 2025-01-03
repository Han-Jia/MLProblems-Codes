import numpy as np


def transform_feature(x1, x2):
    '''
    :参数 x1: 第一个样本
    :参数 x2: 第二个样本
    :返回: 转化后的样本对特征
    该方法将样本对转化为特征线性分类器的特征
    '''
    # 计算样本之间的差
    diff = np.expand_dims(x1 - x2, axis=1)
    # 计算差的外积
    return (diff * diff.T).reshape(-1)


def transform_tuple(tuples, y):
    '''
    :参数 tuples: 样本对, 形状为(n,2,d), n为样本对数量, d为样本的维度
    :参数 y: 样本对的标记, 形状为(n,), 1表示正样本对, 0表示负样本对
    :返回 tuple_X : 转化后的样本对特征, 形状为(n, d^2)
    :返回 tuple_y : 转化后的样本对标记, 形状为(n,)
    '''
    # 将原始的二元组转化使用于线性分类器的特征
    tuple_X = np.array([transform_feature(x1, x2) for x1, x2 in tuples])
    # 将正负样本对的标记颠倒
    tuple_y = 1 - y
    return tuple_X, tuple_y


def transform_triplet(triplets):
    '''
    :参数 triplets: 三元组, 形状为(n,3,d), n为样本对数量, d为样本的维度
    :返回 triplet_X : 转化后的三元组特征, 形状为(2n,d^2)
    :返回 triplet_y : 转化后的三元组标记, 形状为(2n,)
    '''
    # 将原始的三元组转化使用于线性分类器的特征
    triplet_origin_X = np.array([transform_feature(x1, x2) - transform_feature(x1, x3) for x1, x2, x3 in triplets])
    # 为了正常训练一个二分类器, 我们将(a,p,n)作为负类, (a,n,p)作为正类
    triplet_X = np.concatenate([triplet_origin_X, -triplet_origin_X])
    num_triplets = triplet_origin_X.shape[0]
    triplet_y = np.array([0] * num_triplets + [1] * num_triplets)
    return triplet_X, triplet_y
