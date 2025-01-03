import numpy as np

from scipy.stats import multivariate_normal

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

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


def erase_data(X):
    ''' 随机删除某些样本的某些特征
    :参数 X: np.ndarray, 样本特征, 形状为 (N, D)
    :返回: X_erased: np.ndarray, 删除数据后的样本特征, 形状为 (N, D)
    '''

    N = X.shape[0]
    D = X.shape[1]
    X_erased = X.copy()

    for i in range(N):
        for d in range(D):
            # 以0.2的概率删除某个值
            if np.random.rand() < 0.2:
                X_erased[i, d] = np.nan

    return X_erased


def mean_impute(X):
    ''' 使用特征的均值填充缺失值
    :参数 X: np.ndarray, 样本特征, 形状为 (N, D)
    :返回: X_imputed: np.ndarray, 填充缺失值后的样本特征, 形状为 (N, D)
    '''

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)

    return X_imputed


def log_likelihood(X, mu, sigma):
    ''' 计算混合模型的对数似然函数值
    :参数 X: np.ndarray, 填充后的样本特征, 形状为 (N, D)
    :参数 mu: np.ndarray, 高斯模型的均值, 形状为 (D,)
    :参数 sigma: np.ndarray, 高斯模型的协方差矩阵, 形状为 (D, D)
    :返回: log_likelihood: float, 对数似然函数值
    '''

    N = X.shape[0]
    log_likelihood = 0.0

    for i in range(N):
        log_likelihood += np.log(multivariate_normal.pdf(X[i], mu, sigma))

    return log_likelihood


def em_impute(X):
    ''' 使用EM算法填充缺失值
    :参数 X: np.ndarray, 样本特征, 形状为 (N, D)
    :返回: X_imputed: np.ndarray, 填充缺失值后的样本特征, 形状为 (N, D)
    :返回: mu: np.ndarray, 高斯模型的均值, 形状为 (D,)
    :返回: sigma: np.ndarray, 高斯模型的协方差矩阵, 形状为 (D, D)
    :返回: log_likelihood_list: float, 对数似然函数值列表
    '''

    N = X.shape[0]
    D = X.shape[1]

    # 初始化高斯模型的参数
    mu = np.random.rand(D)  # 这里也可以使用其它方式初始化均值
    sigma = np.eye(D)  # 这里也可以使用其它方式初始化协方差矩阵
    # 检查初始化的协方差矩阵是否是正定矩阵
    assert (np.all(np.linalg.eigvals(sigma) > 0))

    log_likelihood_list = []

    converge = False
    iter = 0
    while not converge:
        # 执行E步, 返回各个样本的期望值E_x和各个样本平方的期望值E_xx
        E_x, E_xx = e_step(X, mu, sigma)

        # 执行M步, 返回最新的高斯模型参数
        mu_new, sigma_new = m_step(E_x, E_xx)

        # 检查模型是否收敛
        converge = np.allclose(mu, mu_new) and np.allclose(sigma, sigma_new)

        iter += 1
        print('iter %d finish, log_likelihood = %f' % (iter, log_likelihood(E_x, mu_new, sigma_new)))
        log_likelihood_list.append(log_likelihood(E_x, mu_new, sigma_new))

        # 更新参数
        mu = mu_new
        sigma = sigma_new

    return E_x, mu, sigma, log_likelihood_list


def e_step(X, mu, sigma):
    ''' 执行E步, 返回各个样本的期望值all_E_x和各个样本平方的期望值all_E_xx
    :参数 X: np.ndarray, 填充后的样本特征, 形状为 (N, D)
    :参数 mu: np.ndarray, 高斯模型的均值, 形状为 (D,)
    :参数 sigma: np.ndarray, 高斯模型的协方差矩阵, 形状为 (D, D)
    :返回: E_x: np.ndarray, 各个样本的期望值, 形状为 (N, D)
    :返回: E_xx: np.ndarray, 各个样本平方的期望值, 形状为 (N, D, D)
    '''

    N = X.shape[0]
    D = X.shape[1]

    E_x = np.zeros((N, D))
    E_xx = np.zeros((N, D, D))

    for i in range(N):
        # 若第i个样本中存在缺失值
        if np.isnan(X[i]).any():
            # 分别获取观测值维度和缺失值维度
            # 下标s表示seen, 即观测值, 下标u表示unseen, 即缺失值
            id_s = np.where(~np.isnan(X[i]))[0]
            id_u = np.where(np.isnan(X[i]))[0]

            # 将模型参数按照缺失值维度分块
            mu_s = mu[id_s]
            mu_u = mu[id_u]

            sigma_ss = sigma[id_s, :][:, id_s]
            sigma_su = sigma[id_s, :][:, id_u]
            sigma_us = sigma[id_u, :][:, id_s]
            sigma_uu = sigma[id_u, :][:, id_u]

            # 计算隐变量后验分布的参数
            mu_u_given_s = mu_u + sigma_us @ np.linalg.inv(sigma_ss) @ (X[i][id_s] - mu_s)
            sigma_u_given_s = sigma_uu - sigma_us @ np.linalg.inv(sigma_ss) @ sigma_su
            # 检查条件协方差矩阵是否是正定矩阵
            assert (np.all(np.linalg.eigvals(sigma_u_given_s) > 0))

            # 计算该样本的期望值
            E_x[i, id_s] = X[i][id_s]
            E_x[i, id_u] = mu_u_given_s

            # 计算该样本平方的期望值
            # 使用np.ix_对矩阵分块赋值
            E_xx[i][np.ix_(id_s, id_s)] = np.outer(X[i, id_s], X[i, id_s].T)
            E_xx[i][np.ix_(id_s, id_u)] = np.outer(X[i, id_s], mu_u_given_s.T)
            E_xx[i][np.ix_(id_u, id_s)] = np.outer(mu_u_given_s, X[i, id_s].T)
            E_xx[i][np.ix_(id_u, id_u)] = np.outer(mu_u_given_s, mu_u_given_s.T) + sigma_u_given_s

        # 若第i个样本中不存在缺失值
        else:
            E_x[i] = X[i]
            E_xx[i] = np.outer(X[i], X[i].T)

    return E_x, E_xx


def m_step(E_x, E_xx):
    ''' 执行M步, 返回最新的高斯模型参数
    :参数 E_x: np.ndarray, 各个样本的期望值, 形状为 (N, D)
    :参数 E_xx: np.ndarray, 各个样本平方的期望值, 形状为 (N, D, D)
    :返回: mu: np.ndarray, 高斯模型的均值, 形状为 (D,)
    :返回: sigma: np.ndarray, 高斯模型的协方差矩阵, 形状为 (D, D)
    '''

    N = E_x.shape[0]
    D = E_x.shape[1]

    mu = np.sum(E_x, axis=0) / N

    sigma = np.sum(E_xx, axis=0) / N - np.outer(mu, mu.T)

    # 检查协方差矩阵是否正定
    assert (np.all(np.linalg.eigvals(sigma) > 0))

    return mu, sigma


def draw_log_likelihood_list(log_likelihood_list):
    fig = plt.figure(figsize=(15, 12))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('iteration', fontsize=42)
    ax.set_xlim(0, 30)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': 42})
    ax.set_ylabel('log likelihood', fontsize=42)
    ax.set_ylim(-650, -200)
    ax.set_yticks([-650, -550, -450, -350, -250])
    ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize': 42})

    ax.spines['top'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)

    X = np.arange(len(log_likelihood_list))
    Y = log_likelihood_list
    plt.plot(X, Y, color='#0072BD', linewidth=5)

    plt.tight_layout()
    plt.show()


def train(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy: %f' % (np.mean(y_pred == y_test)))


if __name__ == '__main__':
    np.random.seed(648)

    X, y = read_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=648)
    # print(X_train.shape) # (105, 4)
    # print(X_test.shape) # (45, 4)
    # print(y_train.shape) # (105,)
    # print(y_test.shape) # (45,)

    X_train_erased = erase_data(X_train)

    X_train_imputed_mean = mean_impute(X_train_erased)

    X_train_imputed_em, mu, sigma, log_likelihood_list = em_impute(X_train_erased)

    # draw_log_likelihood_list(log_likelihood_list)

    train(X_train, y_train, X_test, y_test)
    train(X_train_imputed_mean, y_train, X_test, y_test)
    train(X_train_imputed_em, y_train, X_test, y_test)
