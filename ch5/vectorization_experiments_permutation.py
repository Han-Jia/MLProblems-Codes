import time
import numpy as np
import matplotlib.pyplot as plt
from vectorization_plain_permutation import plain_permutation_function
from vectorization_vec_permutation_sparse import vectorized_permutation_sparse_function

size_list = list(range(5000, 100001, 5000))
plain_time_list = []
vec_time_list = []
for size in size_list:
    # 生成随机数据X, X的维度为(size, 10)
    X = np.random.randn(size, 10)
    # 生成随机排列perm
    perm = np.random.permutation(size)
    t1 = time.time()
    # 运行plain实现并计时
    D_plain = plain_permutation_function(X, perm)
    t2 = time.time()
    # 运行vectorized实现并计时
    D_vec = vectorized_permutation_sparse_function(X, perm)
    t3 = time.time()
    # 计算时间
    plain_time_list.append(t2 - t1)
    vec_time_list.append(t3 - t2)
    # 检查两种实现是否一致
    assert np.allclose(D_plain, D_vec, rtol=1e-3)

plt.plot(size_list, plain_time_list, label='plain')
plt.plot(size_list, vec_time_list, label='vectorized_sparse')
plt.legend()
plt.savefig('timing_permutation.png')
