import cvxpy as cp
from sklearn.datasets import make_classification

m = 1000
r = 1.
d = int(m * r)

# 创建一个包含m个样本，每个样本特征维度为d的二分类数据集
X, y = make_classification(m, d)
# 将标记从{0,1}映射至{-1,1}
y = (y * 2 - 1)


# 解决SVM原问题
def solve_primal(X, y, C):
    '''
    :参数 X: ndarray, 形状为(m, d), 样例矩阵
    :参数 y: ndarray, 形状为(m), 样例标签向量
    :参数 C: 标量, 含义与式(6.29)中C相同
    :返回: w, b, SVM的权重与偏置
    '''
    m, d = X.shape
    y_ = y.reshape(-1, 1)
    # 定义CVXPY优化问题的变量w(权重),b(偏置),xi(软间隔)
    w = cp.Variable((d, 1))
    b = cp.Variable()
    xi = cp.Variable((m, 1))
    # 计算loss(经验风险)与reg(结构风险)
    loss = cp.sum(xi)
    reg = cp.sum_squares(w)
    # 定义CVXPY优化问题
    prob = cp.Problem(
        # 目标函数
        cp.Minimize(0.5 * reg + C * loss),
        # 约束             
        [cp.multiply(y_, X @ w + b) >= 1 - xi,
         xi >= 0])
    # 调用CVXPY求解器求解
    prob.solve()
    # 返回求解后的变量
    return w, b


# 解决SVM对偶问题
def solve_dual(X, y, C):
    '''
    :参数 X: ndarray, 形状为(m, d), 样例矩阵
    :参数 y: ndarray, 形状为(m), 样例标签向量
    :参数 C: 标量, 含义与式(6.29)中C相同
    :返回: alpha，SVM的对偶变量
    '''
    pass
