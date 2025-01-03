import copy
import numpy as np
from tqdm import tqdm


class SAMME():
    def __init__(self, base_estimator, n_estimators, n_classes):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_classes = n_classes
        self.onehot_mat = np.diag(np.ones(n_classes))

    def fit(self, X, y, init_weight=None):
        m, d = X.shape

        # initialize sample weights
        if init_weight is not None:
            assert len(init_weight) == m
        else:
            init_weight = np.ones(m) / m

        # base estimators & its weights
        self.alphas = []
        self.estimators = []

        weight = copy.deepcopy(init_weight)

        for t in tqdm(range(self.n_estimators)):
            # train base estimator with sample_weight as weight
            clf = copy.deepcopy(self.base_estimator)
            clf.fit(X, y, sample_weight=weight)

            # return hard label
            pred_y = clf.predict(X)

            # calculate this estimator's weights: have to multiply ``weight"
            err = np.mean((pred_y != y).astype(np.float) * weight)

            # stop condition
            if 1.0 - err <= 1.0 / self.n_classes:
                break

            # calculate estimator's weight
            alpha = np.log((self.n_classes - 1) * (1.0 - err) / err)

            self.alphas.append(alpha)
            self.estimators.append(clf)

            # update samples' weight
            weight = weight * np.exp(alpha * (pred_y != y).astype(np.float))
            weight = weight / np.sum(weight)

    def predict(self, X):
        m = X.shape[0]

        preds = np.zeros((m, self.n_classes))

        # sequential prediction
        for t in range(len(self.estimators)):
            pred = self.estimators[t].predict(X)
            pred_onehot = self.onehot_mat[pred]
            preds += self.alphas[t] * pred_onehot

        pred_y = np.argmax(preds, axis=1)
        return pred_y

    def report(self, X, y):
        """ predict sequential errors via each stage's prediction
        """
        m = X.shape[0]

        preds = np.zeros((m, self.n_classes))

        errs = []
        for t in range(len(self.estimators)):
            pred = self.estimators[t].predict(X)
            pred_onehot = self.onehot_mat[pred]
            preds += self.alphas[t] * pred_onehot

            err = np.mean(np.argmax(preds, axis=1) != y)
            errs.append(err)

        return errs


class SAMMER():
    def __init__(self, base_estimator, n_estimators, n_classes):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_classes = n_classes
        self.onehot_mat = np.diag(np.ones(n_classes))

    def fit(self, X, y, init_weight=None):
        m, d = X.shape
        C = self.n_classes

        # initialize sample weights
        if init_weight is not None:
            assert len(init_weight) == m
        else:
            init_weight = np.ones(m) / m

        # base estimators
        self.estimators = []

        weight = copy.deepcopy(init_weight)

        for t in tqdm(range(self.n_estimators)):
            # train base estimator with sample_weight as weight
            clf = copy.deepcopy(self.base_estimator)
            clf.fit(X, y, sample_weight=weight)

            # return soft-label
            pred_prob = clf.predict_proba(X)

            self.estimators.append(clf)

            # update samples' weight
            # mat_y[i] = [..., -1.0 / (C - 1), 1.0, -1.0 / (C - 1), ...]
            mat_y = -1.0 / (C - 1.0) * np.ones((m, C))
            mat_y[np.arange(m), y] = 1.0

            # y^T log p(x)
            prod = (mat_y * np.log(pred_prob + 1e-8)).sum(axis=1)
            weight = weight * np.exp(-1.0 * (1.0 - 1.0 / C) * prod)
            weight = weight / np.sum(weight)

    def predict(self, X):
        m = X.shape[0]
        C = self.n_classes

        pred_probs = np.zeros((m, C))

        # sequential prediction
        for t in range(len(self.estimators)):
            pred_prob = self.estimators[t].predict_proba(X)

            log_pred = np.log(pred_prob + 1e-8)
            log_pred = (log_pred - log_pred.mean(axis=1, keepdims=True))
            pred_probs += (C - 1.0) * log_pred

        pred_y = np.argmax(pred_probs, axis=1)
        return pred_y

    def report(self, X, y):
        """ predict sequential errors via each stage's prediction
        """
        m = X.shape[0]
        C = self.n_classes

        pred_probs = np.zeros((m, C))
        errs = []

        # sequential prediction
        for t in range(len(self.estimators)):
            pred_prob = self.estimators[t].predict_proba(X)

            log_pred = np.log(pred_prob + 1e-8)
            log_pred = (log_pred - log_pred.mean(axis=1, keepdims=True))
            pred_probs += (C - 1.0) * log_pred

            err = np.mean(np.argmax(pred_probs, axis=1) != y)
            errs.append(err)

        return errs
