import numpy as np

from scipy.stats import multivariate_normal

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import normalized_mutual_info_score

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'


def read_dataset():
    iris = load_iris()
    X = iris.data  # 样本特征, 形状为 (150, 4)
    y = iris.target  # 样本标记, 形状为 (150,)

    return X, y


def standardize(X):
    # 原始特征均值为[5.8433, 3.0573, 3.7580, 1.1993]
    # print(np.mean(X, axis=0))
    # 原始特征标准差为[0.8253, 0.4344, 1.7594, 0.7596]
    # print(np.std(X, axis=0))

    scaler = StandardScaler()
    standardized_X = scaler.fit_transform(X)
    # 标准化后特征均值为[0.0, 0.0, 0.0, 0.0]
    # print(np.mean(standardized_X, axis=0))
    # 标准化后特征标准差为[1.0, 1.0, 1.0, 1.0]
    # print(np.std(standardized_X, axis=0))

    return standardized_X


def normalize(X):
    normalizer = Normalizer()
    normalized_X = normalizer.fit_transform(X)

    return normalized_X


def observed_data_log_likelihood(X, pi, mu, sigma):
    ''' 计算观测数据的对数似然函数值
    :参数 X: np.ndarray, 样本特征, 形状为 (N, D)
    :参数 pi: np.ndarray, 混合模型的分布概率, 形状为 (K,)
    :参数 mu: np.ndarray, 混合模型的均值, 形状为 (K, D) 
    :参数 sigma: np.ndarray, 混合模型的方差, 形状为 (K, D, D)
    :返回: log_likelihood: float, 数据的对数似然函数值
    '''

    N = X.shape[0]
    K = pi.shape[0]
    log_likelihood = 0.0

    for i in range(N):
        tmp = 0.0
        for k in range(K):
            tmp += pi[k] * multivariate_normal.pdf(X[i], mu[k], sigma[k])
        log_likelihood += np.log(tmp)

    return log_likelihood


def build_gmm(X):
    ''' 基于EM算法, 使用高斯混合模型对数据$\mX$建模
    :参数 X: np.ndarray, 样本特征, 形状为 (N, D)
    :返回: pi: np.ndarray, 混合模型的分布概率, 形状为 (K,)
    :返回: mu: np.ndarray, 混合模型的均值, 形状为 (K, D) 
    :返回: sigma: np.ndarray, 混合模型的方差, 形状为 (K, D, D)
    :返回: observed_data_log_likelihood_list: list, 观测数据的对数似然函数值序列
    '''

    # 设置混合成分个数为3
    K = 3
    # 初始化参数
    pi = np.random.rand(K)
    pi = pi / np.sum(pi)

    mu = np.random.rand(K, X.shape[1])

    sigma = np.identity(np.shape(X)[1])
    sigma = np.tile(sigma, (K, 1, 1))

    # 检查初始化的协方差矩阵是否正定
    for i in range(0, K):
        flag = np.all(np.linalg.eigvals(sigma[i]) > 0)
        assert (flag)

    # 初始化观测数据的对数似然
    observed_data_log_likelihood_list = []

    converge = False
    iter = 0
    while not converge:
        # 执行E步, 返回隐变量的期望值Z
        Z = gmm_e_step(X, pi, mu, sigma)
        # 执行M步, 获取新参数pi, mu, sigma
        pi_new, mu_new, sigma_new = gmm_m_step(X, Z, pi, mu, sigma)

        # 检查模型是否收敛
        converge = np.allclose(pi, pi_new) and \
                   np.allclose(mu, mu_new) and \
                   np.allclose(sigma, sigma_new)

        iter += 1
        print('iter %d finish, observed_data_log_likelihood = %f' %
              (iter, observed_data_log_likelihood(X, pi_new, mu_new, sigma_new)))

        observed_data_log_likelihood_list.append(observed_data_log_likelihood(X, pi_new, mu_new, sigma_new))

        # 更新参数
        pi = pi_new
        mu = mu_new
        sigma = sigma_new

    return pi, mu, sigma, observed_data_log_likelihood_list, Z


def gmm_e_step(X, pi, mu, sigma):
    ''' 适用于高斯混合模型的E步
    :参数 X: np.ndarray, 样本特征, 形状为 (N, D)
    :参数 pi: np.ndarray, 混合模型的分布概率, 形状为 (K,)
    :参数 mu: np.ndarray, 混合模型的均值, 形状为 (K, D)
    :参数 sigma: np.ndarray, 混合模型的方差, 形状为 (K, D, D)
    :返回: Z: np.ndarray, 隐变量的期望值, 形状为 (N, K)
    '''

    N = X.shape[0]
    K = pi.shape[0]

    Z = np.zeros((N, K))

    for i in range(N):
        for k in range(K):
            Z[i, k] = pi[k] * multivariate_normal.pdf(X[i], mu[k], sigma[k])

    # 隐变量归一化
    Z = Z / np.sum(Z, axis=1, keepdims=True)

    return Z


def gmm_m_step(X, Z, pi, mu, sigma):
    ''' 适用于高斯混合模型的M步
    :参数 X: np.ndarray, 样本特征, 形状为 (N, D)
    :参数 Z: np.ndarray, 隐变量的期望值, 形状为 (N, K)
    :参数 pi: np.ndarray, 混合模型的分布概率, 形状为 (K,)
    :参数 mu: np.ndarray, 混合模型的均值, 形状为 (K, D)
    :参数 sigma: np.ndarray, 混合模型的方差, 形状为 (K, D, D)
    :返回: pi_new: np.ndarray, 混合模型的分布概率, 形状为 (K,)
    :返回: mu_new: np.ndarray, 混合模型的均值, 形状为 (K, D) 
    :返回: sigma_new: np.ndarray, 混合模型的方差, 形状为 (K, D, D)
    '''

    N = X.shape[0]
    K = pi.shape[0]

    # 更新分布概率
    pi_new = np.sum(Z, axis=0) / N

    # 更新均值
    mu_new = np.zeros((K, X.shape[1]))
    for k in range(K):
        mu_new[k] = np.sum(Z[:, k].reshape(-1, 1) * X, axis=0) / np.sum(Z[:, k])

    # 更新协方差矩阵
    sigma_new = np.zeros((K, X.shape[1], X.shape[1]))
    for k in range(K):
        A = (Z[:, k].reshape(-1, 1) * (X - mu_new[k])).transpose((1, 0))
        B = X - mu_new[k]
        sigma_new[k] = np.matmul(A, B) / np.sum(Z[:, k])

    # 检查更新后的协方差矩阵是否正定
    for k in range(K):
        flag = np.all(np.linalg.eigvals(sigma_new[k]) > 0)
        assert (flag)

    return pi_new, mu_new, sigma_new


def draw_observed_data_log_likelihood_list(observed_data_log_likelihood_list):
    fig = plt.figure(figsize=(15, 12))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('iteration', fontsize=42)
    ax.set_xlim(0, 35)
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': 42})
    ax.set_ylabel('log likelihood of observed data', fontsize=42)
    ax.set_ylim(-400, -200)
    ax.set_yticks([-400, -350, -300, -250, -200])
    ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize': 42})

    ax.spines['top'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)

    X = np.arange(len(observed_data_log_likelihood_list))
    Y = observed_data_log_likelihood_list
    plt.plot(X, Y, color='#0072BD', linewidth=5)

    plt.tight_layout()
    plt.show()


def get_cluster(Z, y):
    ''' 适用于高斯混合模型的聚类
    :参数 Z: np.ndarray, 隐变量的期望值, 形状为 (N, K)
    :参数 y: np.ndarray, 样本标签, 形状为 (N,)
    :返回: cluster: np.ndarray, 聚类结果, 形状为 (N,)
    '''

    cluster = np.argmax(Z, axis=1)
    print(cluster)
    print(normalized_mutual_info_score(cluster, y))

    return cluster


if __name__ == '__main__':
    # np.random.seed(648)

    X, y = read_dataset()
    # print(X)
    # print(y)

    standardized_X = standardize(X)
    normalized_X = normalize(X)

    pi, mu, sigma, observed_data_log_likelihood_list, Z = build_gmm(X)
    # pi, mu, sigma, observed_data_log_likelihood_list = build_gmm(standardized_X)
    # pi, mu, sigma, observed_data_log_likelihood_list = build_gmm(normalized_X)

    # draw_observed_data_log_likelihood_list(observed_data_log_likelihood_list)

    get_cluster(Z, y)
