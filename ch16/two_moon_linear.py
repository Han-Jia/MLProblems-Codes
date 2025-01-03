from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 创建线性分类器LogisticRegression
lr = LogisticRegression()
# 投入训练数据
lr.fit(X_train, y_train)
# 计算在测试数据上的准确率
score = lr.score(X_test, y_test)
# k近邻分类器可以处理非线性数据
knn = KNeighborsClassifier()
