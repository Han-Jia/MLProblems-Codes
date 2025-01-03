from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 产生一个没有冗余特征的二分类数据集
X, y = make_classification(n_samples=500,  # 生成一个包含500个样本的二维特征二分类数据集
                           n_features=2,  # n_features指定特征数为2
                           n_redundant=0,  # n_redundant指定冗余特征数
                           n_informative=2,  # n_informative指定了有效特征数
                           n_clusters_per_class=1,  # n_clusters_per_class指定了每个类中有多少个簇
                           random_state=0)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
