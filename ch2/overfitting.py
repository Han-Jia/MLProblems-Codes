import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

'''
Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
'''


def true_fun(X):
    return np.cos(1.5 * np.pi * X + 0.5 * np.pi)


np.random.seed(0)

n_samples = 50
degrees = [1, 4, 30]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

matplotlib.rc('axes', linewidth=2)

for i in range(len(degrees)):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(
        degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    # scores = cross_val_score(
    #     pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10
    # )

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), 'b-',
             linewidth=2, label="Model")
    # plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, c="k", s=30, label="Samples")
    plt.xlabel("$x$", fontsize=30)
    plt.ylabel("$y$", fontsize=30)
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    # plt.legend(loc="best")
    # plt.title(
    #     "Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
    #         degrees[i], -scores.mean(), scores.std()
    #     )
    # )
    plt.savefig('ch2_overfit_{}-degree.pdf'.format(degrees[i]))
