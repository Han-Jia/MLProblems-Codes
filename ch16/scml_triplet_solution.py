from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def create_triplet(X, y, num_pos=-1, num_neg=-1, random_pos=False, random_neg=False):
    '''
    :参数 X: 样例矩阵，形状为(m,d)，m为样本数量，d为特征维度
    :参数 y: 样例的标记，形状为(m,)
    :参数 num_pos: 每个样例抽取的正样本数量，-1代表抽取所有同类样本
    :参数 num_neg: 每个样例抽取的负样本数量，-1代表抽取所有异类样本
    :参数 random_pos: 随机抽取正样本，False代表抽取近邻同类样例作为正样本
    :参数 random_neg: 随机抽取负样本，False代表抽取近邻异类样例作为正样本
    :返回: triplets: 三元组，其形状为(n,3)，n代表三元组的数量，由m, num_pos, num_neg共同决定，
    三元组的每一个元素代表样例的索引
    '''
    # 通过样例标记将每个类的样例索引放到单独的数组里，第一个数组存放了第一个类的样例索引，
    # 第二个数组存放了第二个类的样例索引，以此类推
    unique_y = np.unique(y)
    cls_idxes = []
    arange = np.arange(X.shape[0])
    for l in unique_y:
        cls_idx = arange[y == l]
        cls_idxes.append(cls_idx)
    # 计算样本之间的距离
    distances = euclidean_distances(X)
    # 将与自身的距离设为无穷，避免选择到样例自身
    for i in range(X.shape[0]):
        distances[i, i] = np.inf
    triplets = []
    # 遍历每个类的样本，选取正负样本
    for cls, cls_idx in enumerate(cls_idxes):
        all_idxes = np.arange(X.shape[0])
        bool_pos_idx = np.zeros(X.shape[0], dtype=bool)
        bool_pos_idx[cls_idx] = True
        ncls_idx = all_idxes[~bool_pos_idx]
        for i, a in enumerate(cls_idx):
            dists = distances[a]
            # 获得当前样本和同类、异类样本间的距离
            pos_dists = dists[bool_pos_idx]
            neg_dists = dists[~bool_pos_idx]
            # 对距离进行排序
            sorted_pos_idx = np.argsort(pos_dists)
            sorted_neg_idx = np.argsort(neg_dists)
            if random_pos and num_pos > 0:
                # 随机从同类样本中抽取正样本
                pos_idx = np.random.choice(np.delete(cls_idx, i), num_pos, replace=False)
            else:
                # 抽取num_pos个最近的同类样本
                pos_idx = cls_idx[sorted_pos_idx[:num_pos]]
            if random_neg and num_neg > 0:
                neg_idx = np.random.choice(ncls_idx, num_neg, replace=False)
            else:
                neg_idx = ncls_idx[sorted_neg_idx[:num_neg]]
            # 将索引加入三元组
            for p in pos_idx:
                for n in neg_idx:
                    triplets.append([a, p, n])
    return triplets
