# @author : Administrator 
# @date : 2022/4/21
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.datasets import make_swiss_roll
from collections import OrderedDict
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from matplotlib.ticker import NullFormatter

import matplotlib
from matplotlib import pyplot as plt

config = {
    'font.family': 'serif',
    'font.serif': ['kaiti'],
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
    'font.size': 12,
}
plt.rcParams.update(config)

matplotlib.rc('axes', linewidth=2)

# 生成S型流型, X代表流型上的点的三维坐标, y代表该点在流型上的"位置"
X, y = make_swiss_roll(1000, random_state=0)

ax = plt.subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
ax.view_init(4, -72)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())
plt.savefig(os.path.join('figures', f'ch16_swiss_roll.pdf'), transparent=True, bbox_inches='tight')
plt.close()

os.makedirs('figures', exist_ok=True)

methods = OrderedDict()
methods['PCA'] = PCA(n_components=2)
methods['LLE'] = LocallyLinearEmbedding(n_neighbors=12, n_components=2)
for label, method in methods.items():
    Y = method.fit_transform(X)
    ax = plt.subplot()
    ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
    ax.set_title(f"{label}", font='sans serif')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis("tight")
    plt.savefig(os.path.join('figures', f'ch16_swiss_roll_{label}.pdf'), transparent=True, bbox_inches='tight')
    plt.close()
