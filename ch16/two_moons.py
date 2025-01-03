from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 生成双月数据
X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
