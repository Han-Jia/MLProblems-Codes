from perceptron_train import train
from perceptron_test import test
from perceptron_train_fn import X, y_and, y_or, y_xor

# 学习或函数, 很快收敛, 4个样本都分类正确
w_or = train(X, y_or, eta=0.1, epoch=100)
or_correct_count = test(X, y_or, w_or)
print("or_correct_count:", or_correct_count)

# 学习与函数, 很快收敛, 4个样本都分类正确
w_and = train(X, y_and, eta=0.1, epoch=100)
and_correct_count = test(X, y_and, w_and)
print("and_correct_count:", and_correct_count)

# 学习异或函数, 增大训练轮数epoch也不能收敛
# 因为不能收敛，返回的w_xor是None
w_xor = train(X, y_xor, eta=0.1, epoch=10000)
xor_correct_count = test(X, y_xor, w_xor)
print("xor_correct_count:", xor_correct_count)
