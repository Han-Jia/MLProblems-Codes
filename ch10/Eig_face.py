# yale_face_recognition.py
import scipy.io as scio
from numpy import linalg
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'serif',
        'weight': 'normal',
        'size': 14,
        }

label_font = {'family': 'serif',
              'weight': 'normal',
              'size': 12,
              }


# 读取Yale_Face数据
def load_dataset():
    data_name = "Yale_64x64.mat"
    data = scio.loadmat(data_name)
    return data


# 对样本进行中心化
def center_face(face):
    mean_face = np.mean(face, axis=0)
    centered_face = face - mean_face
    return centered_face, mean_face


# 将样本划分成训练集和验证集
def split_data(norm_face, split_ratio=0.8):
    data_num = norm_face.shape[0]
    data_idx = np.arange(data_num)
    np.random.shuffle(data_idx)
    shuffle_data = norm_face[data_idx]
    train_num = int(data_num * split_ratio)
    train_data = shuffle_data[:train_num, :]
    val_data = shuffle_data[train_num:, :]
    return train_data, val_data


# 通过对X^TX进行特征分解间接计算特征值和特征向量
def eig_val(centered_face):
    cov_mat = (np.dot(centered_face, centered_face.T))
    eig_values, eig_vectors = linalg.eig(cov_mat)
    eig_vectors = np.dot(centered_face.T, eig_vectors)
    eig_norm = np.linalg.norm(eig_vectors, ord=2, axis=0, keepdims=True)
    eig_vectors = eig_vectors / eig_norm

    return eig_values.astype(float), eig_vectors.astype(float)


# 直接对XX^T进行特征分解计算特征值和特征向量
def eig_val_2(centered_face):
    cov_mat = (np.dot(centered_face.T, centered_face))
    eig_values, eig_vectors = linalg.eig(cov_mat)
    return eig_values.astype(float), eig_vectors.astype(float)


# 使用SVD计算奇异值和奇异向量
def svd(centered_face):
    U, Sigma, Vt = np.linalg.svd(centered_face, full_matrices=False, compute_uv=True)
    return Sigma, Vt


# 计算特征脸
def pca(eig_values, eig_vectors, k):
    eig_idx = np.argsort(-eig_values)[:k]
    eig_faces = eig_vectors[:, eig_idx]
    return eig_faces, eig_vec


def plot_eig_face(eig_values, eig_faces, mean_vector, eig_face_num=10):
    eig_idx = np.argsort(-eig_values)[:eig_face_num]
    eig_faces = eig_faces[:, eig_idx]
    fig, axs = plt.subplots(2, 5, constrained_layout=True, figsize=(12, 6.4))
    mean_vector = mean_vector.reshape(64, 64)
    eig_f = np.reshape(eig_faces.T, (10, 64, 64))

    def sub_plot(ax, i):
        ax.imshow(np.rot90(eig_f[i], -1), cmap=plt.get_cmap("gray"))
        # ax.text(0, -0.5, f"eig_face-{i}", size=15, ha="center")
        ax.set_title(f"Eigenface-{i + 1}", font)
        ax.axis('off')

    for ax, i in zip(axs.flat, range(10)):
        sub_plot(ax, i)
    # plt.show()
    plt.savefig("eig_face.pdf")


def plot_pca_face(norm_face, eig_vec, mean_face):
    selected_face = norm_face[[1, 50, 100], :]
    weight = np.dot(selected_face, eig_vec).T
    inver_face = []
    eig_list = [1, 5, 10, 20, 40, 80]
    for e in [1, 5, 10, 20, 40, 80]:
        inver_face.append(np.dot(eig_vec[:, :e], weight[:e, :]).T)
    fig, axs = plt.subplots(3, 7, constrained_layout=True, figsize=(15, 6.4))
    for i in range(3):
        origin_face = selected_face[i]
        ax = axs[i][0]
        ax.imshow(np.rot90((origin_face + mean_face).reshape(64, 64), -1), cmap=plt.get_cmap("gray"))
        ax.set_title(f"OriginFace-{i + 1}")
        ax.axis('off')
        for j in range(6):
            ax = axs[i][j + 1]
            ax.imshow(np.rot90((inver_face[j][i] + mean_face).reshape(64, 64), -1), cmap=plt.get_cmap("gray"))
            ax.set_title(f"EigNum-{eig_list[j]}", font)
            ax.axis('off')
    # plt.show()
    plt.savefig("pca_face.pdf")


def plot_eig_value(eig_value):
    eig_value = (-np.sort(-eig_value)).tolist()
    eig_ratio_list = []
    cumu_eig = 0
    sum_eig = sum(eig_value)
    for i in eig_value:
        cumu_eig += i
        eig_ratio_list.append(cumu_eig / sum_eig * 100)
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(6.4, 5))
    ax = axs[0]
    ax.plot(range(len(eig_value)), eig_value)
    ax.set_title("Eigenvalues", font)
    ax = axs[1]
    ax.plot(range(len(eig_ratio_list)), eig_ratio_list)
    ax.set_title("Proportion of cumulative eigenvalues", font)
    plt.savefig("eig_value.pdf")
    # plt.show()


# 计算不同个数特征脸下重构人脸的rmse
def inv_error(norm_face, Vt):
    error = []
    for k in range(Vt.shape[0]):
        Wt = Vt[:k + 1, :]
        inv_face = np.dot(Wt.T, np.dot(Wt, norm_face.T))
        rmse = np.linalg.norm((norm_face - inv_face.T), ord='fro') / np.sqrt(norm_face.shape[0])
        error.append(rmse)
    return error


# 在训练集上计算降维矩阵，并分别在训练集和验证集上重构，绘制rmse曲线
def plot_rmse(norm_face):
    train_data, val_data = split_data(norm_face)
    sigma, Vt = svd(train_data)
    train_error_list = inv_error(train_data, Vt)
    test_error_list = inv_error(val_data, Vt)
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4.5))
    ax = axs[0]
    ax.plot(range(len(train_error_list)), train_error_list)
    ax.set_title("train set reconstruction error", font)
    ax.set_xlabel('num eigenface', label_font)
    ax.set_ylabel('rmse', label_font)
    ax = axs[1]
    ax.plot(range(len(test_error_list)), test_error_list)
    ax.set_title("test set reconstruction error", font)
    ax.set_xlabel('num eigenface', label_font)
    ax.set_ylabel('rmse', label_font)
    # plt.show()
    plt.savefig("rmse.pdf")


# labels will contain the label that is assigned to the image
if __name__ == '__main__':
    data = load_dataset()
    face_data = data["fea"]
    label = data["gnd"]
    norm_face, mean_face = center_face(face_data)
    eig_value, eig_vec = eig_val(norm_face)
    sigma, Vt = svd(norm_face)
    plot_eig_face(eig_value, eig_vec, mean_face)
    plot_eig_value(eig_value)
    plot_pca_face(norm_face, eig_vec, mean_face)
    plot_rmse(norm_face)
