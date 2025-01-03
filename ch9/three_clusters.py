from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import copy

im = Image.open("./mountain.jpg")
data = im.getdata()
data = np.array(data, dtype='float')

# kmeans
for k in [4, 8, 12, 16]:
    data_kmeans = copy.deepcopy(data)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_kmeans)
    res = kmeans.labels_
    center = kmeans.cluster_centers_
    for i in range(data_kmeans.shape[0]):
        data_kmeans[i] = center[res[i]]
    data_kmeans = np.reshape(data_kmeans, (180, 320, 3), order='C')
    im_res = Image.fromarray(np.uint8(data_kmeans))
    im_res.show()
    im_res.save("./pic_kmeans_k_{}.jpg".format(k))

# DBSCAN
for eps in [1.0, 1.5, 2.0, 2.5]:
    data_dbscan = copy.deepcopy(data)
    db = DBSCAN(eps=eps).fit(data_dbscan)
    indice = db.core_sample_indices_
    label = db.labels_
    components = db.components_
    cr = (label < 0)
    label[cr] = 0
    n_clusters = len(set(label))
    center = np.zeros(n_clusters)
    for j in range(n_clusters):
        for i in range(data_dbscan.shape[0]):
            if label[i] == j:
                center[j] = i
                break
    for i in range(data_dbscan.shape[0]):
        data_dbscan[i] = data_dbscan[int(center[label[i]])]
    data_dbscan = np.reshape(data_dbscan, (180, 320, 3), order='C')
    im_res = Image.fromarray(np.uint8(data_dbscan))
    im_res.show()
    im_res.save("./pic_DBSCAN_eps_{}.jpg".format(eps))

# AGNES
for n_clusters in [4, 8, 12, 16]:
    data_agnes = copy.deepcopy(data)
    ag = AgglomerativeClustering(n_clusters=n_clusters).fit(data_agnes)
    label = ag.labels_
    print(label, label.shape)
    center = np.zeros(n_clusters)
    for j in range(n_clusters):
        for i in range(data_agnes.shape[0]):
            if label[i] == j:
                center[j] = i
                break
    for i in range(data_agnes.shape[0]):
        data_agnes[i] = data_agnes[int(center[label[i]])]
    data_agnes = np.reshape(data_agnes, (180, 320, 3), order='C')
    im_res = Image.fromarray(np.uint8(data_agnes))
    im_res.show()
    im_res.save("./pic_AGNES_numcluster_{}.jpg".format(n_clusters))
