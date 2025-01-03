import numpy as np

np.random.seed(0)
# 随机生成服从标准高斯分布的样本，特征维度为2
X = np.random.randn(300, 2)
# 计算标记，两个特征符号相同时为0不同时为1
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
