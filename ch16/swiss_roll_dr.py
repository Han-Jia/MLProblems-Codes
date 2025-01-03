from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

# PCA取前两个主成分，将数据降至2维
PCA = PCA(n_components=2)
# LLE取每个样本周围12个邻居
LLE = LocallyLinearEmbedding(n_neighbors=12, n_components=2)
