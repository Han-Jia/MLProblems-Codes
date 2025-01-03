from sklearn.neighbors import KNeighborsClassifier

# 创建k=K的k近邻分类器
clf = KNeighborsClassifier(K)
# 分类器训练，对于k近邻分类器而言，训练过程仅仅是将训练数据保存下来
clf.fit(X_train, y_train)
# 计算分类器在测试数据上的精度
test_accuracy = clf.score(X_test, y_test)
