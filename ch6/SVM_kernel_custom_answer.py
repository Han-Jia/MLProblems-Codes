import numpy as np
from sklearn import svm


def custom_kernel(X, Y):
    return 1 / (1 + np.sum((np.expand_dims(X, 1) - np.expand_dims(Y, 0)) ** 2, -1))


clf = svm.SVC(kernel=custom_kernel)
