import time
import numpy as np
import matplotlib.pyplot as plt
from vectorization_plain_distance import plain_distance_function
from vectorization_vec_distance import vectorized_distance_function

size_list = list(range(1, 10, 1)) \
            + list(range(10, 100, 10)) \
            + list(range(100, 1001, 100))
plain_time_list = []
vec_time_list = []
for size in size_list:
    # 生成随机数据X, X的维度为(size, 10)
    X = np.random.randn(size, 10)
    t1 = time.time()
    # 运行plain实现并计时
    D_plain = plain_distance_function(X)
    t2 = time.time()
    # 运行vectorized实现并计时
    D_vec = vectorized_distance_function(X)
    t3 = time.time()
    # 计算运行时间
    plain_time_list.append(t2 - t1)
    vec_time_list.append(t3 - t2)
    # 检查两种实现是否一致
    assert np.allclose(D_plain, D_vec, rtol=1e-3)

plt.plot(size_list, plain_time_list, label='plain')
plt.plot(size_list, vec_time_list, label='vectorized')
plt.legend()
plt.yscale('log')
plt.savefig('timing_distance.png')
