from PIL import Image
import numpy as np


# 绘制聚类后的图像，函数的输入为数据，簇心，数据所属的簇， 图像的宽度和高度， 输出文件名
def plot(data, centroids, cluster_index, width, height, outname):
    for i in range(data.shape[0]):
        data[i] = centroids[cluster_index]  # 将所有像素点的表示替换成所属簇心的表示
    data = np.reshape(data, (height, width, 3), order='C')  # 维度变换，3代表RGB三通道
    im_res = Image.fromarray(np.uint8(data))
    im_res.save(outname)
