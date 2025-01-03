import cvxpy as cp


# 解决SVM对偶问题
def solve_dual(X, y, C):
    m = X.shape[0]
    # 定义对偶问题的优化变量alpha
    alpha = cp.Variable(m)
    y_ = y.reshape(-1, 1)
    # 计算Q矩阵
    Q = y_ * y_.T * (X @ X.T)
    # 计算损失函数，quad_form(x,A)等价于计算x^T Ax的值
    loss = 0.5 * cp.quad_form(alpha, Q) - cp.sum(alpha)
    # 定义优化问题目标和约束
    prob = cp.Problem(cp.Minimize(loss),
                      [cp.sum(cp.multiply(y, alpha)) == 0,
                       alpha >= 0,
                       alpha <= C])
    # 调用求解器求解
    prob.solve()
    return alpha
