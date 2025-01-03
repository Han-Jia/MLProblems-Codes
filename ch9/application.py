from PIL import Image
import numpy as np
from sklearn.cluster import AgglomerativeClustering


# 读取数据，转换成numpy数组的形式，并且输出原始图片的宽度和高度
def load_data(inpath):
    im = Image.open(inpath)
    width, height = im.size
    data = im.getdata()
    data = np.array(data, dtype='float')
    return data, width, height


def plot(data, centroids, cluster_index, width, height, outpath):
    pass


def kmeans(data, n):
    ''' kmeans算法
    :参数 data: 输入数据
    :参数 n: 簇个数
    :返回: 簇中心
    :返回: 样本的簇标签
    '''
    pass


if __name__ == 'main':
    data, width, height = load_data(inpath)
    # 这里以使用kmeans聚类为例，使用其他聚类方法时同理
    centroids, cluster_index = kmeans(data, n)  # 输出簇心的表示，以及所有数据所属的簇
    plot(data, centroids, cluster_index, width, height, outpath)
