from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 读取手写数字数据集
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=0)
