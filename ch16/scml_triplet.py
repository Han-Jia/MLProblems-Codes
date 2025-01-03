from metric_learn import SCML
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 读取威斯康辛州乳腺癌数据集，这是个有569个样本的30维二分类数据集
X, y = load_breast_cancer(return_X_y=True)
# 处于效率考虑，我们取25%的数据
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.25)


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
    ...
    return triplets


# 根据样例标记和采样策略产生三元组
triplets = create_triplet(X_train, y_train, num_pos, num_neg, random_pos, random_neg)
# 构建SCML学习器，在构建时将训练数据传入做预处理
scml = SCML(preprocessor=X_train, random_state=0)
# 根据三元组中的索引训练度量
scml.fit(triplets)
# 对训练集和测试集的特征进行转换
X_tr = scml.transform(X_train)
X_t = scml.transform(X_test)
# 使用k近邻分类器进行评估
knn = KNeighborsClassifier()
knn.fit(X_tr, y_train)
acc = knn.score(X_t, y_test)
