from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 构建一个pipeline，首先将每一维归一化到0均值1方差，然后使用PCA降维到n维
pca = make_pipeline(StandardScaler(), PCA(n_components=n, random_state=0))
# 这里我们选择3NN分类器
knn = KNeighborsClassifier(n_neighbors=3)
# PCA训练，获取降维矩阵
pca.fit(X_train, y_train)
# kNN训练时要首先将样本降维
knn.fit(pca.transform(X_train), y_train)
# kNN测试时也需要先将测试样本降维
acc_knn = knn.score(pca.transform(X_test), y_test)
