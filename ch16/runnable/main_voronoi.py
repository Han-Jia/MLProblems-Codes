from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import os

import matplotlib
from matplotlib import pyplot as plt

config = {
    'font.family': 'serif',
    'font.serif': ['kaiti'],
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
    'font.size': 14,
}
plt.rcParams.update(config)

matplotlib.rc('axes', linewidth=2)


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            mid_point = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(mid_point - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# 生成数据集
n = 20
X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2,
                           random_state=0)

os.makedirs('figures', exist_ok=True)

bg_rgb = matplotlib.colors.ListedColormap([
    # '#F5DEB3','#BDFCC9',
    '#B0E0E6', '#FFC0CB'])
colormap = plt.get_cmap('tab20')
margin = 0.1
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig = plt.figure()

# add 4 distant dummy points

# compute Voronoi texsselation
vor = Voronoi(X)

# plot
regions, vertices = voronoi_finite_polygons_2d(vor)

# colorize
for i, region in enumerate(regions):
    polygon = vertices[region]
    plt.fill(*zip(*polygon), alpha=0.5, c=colormap.colors[i])
for i, region in enumerate(regions):
    polygon = vertices[region]
    plt.plot(*zip(*polygon), c='k')
plt.scatter(X[:, 0], X[:, 1], s=30, marker='.', c='k')
plt.xticks([])
plt.yticks([])
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.savefig(os.path.join('figures', f'ch16_voronoi.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
voronoi = KNeighborsClassifier(n_neighbors=1)
voronoi.fit(X, y)
Z = voronoi.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=bg_rgb)
# plt.contour(xx, yy, Z_voronoi, colors='k')
for i, region in enumerate(regions):
    polygon = vertices[region]
    plt.plot(*zip(*polygon), c='k')
plt.scatter(X[:, 0], X[:, 1], s=30, marker='.', c='k')
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.xticks([])
plt.yticks([])
plt.savefig(os.path.join('figures', f'ch16_nn.pdf'), transparent=True, bbox_inches='tight')
plt.close(fig)
