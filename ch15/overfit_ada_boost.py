import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from ada_boost import SAMME, SAMMER
from Data import load_data


def train_ada_boost(dataset, domain_shift, base="DTC", algorithm="SAMME"):
    # load data
    tr_X, tr_Y, te_X1, te_Y1, te_X2, te_Y2 = load_data(
        dataset=dataset,
        domain_shift=domain_shift
    )
    n_classes = len(np.unique(tr_Y))

    # construct base estimator
    if base == "DTC":
        base_estimator = DecisionTreeClassifier(
            max_depth=1
        )
    elif base == "LR":
        base_estimator = LogisticRegression(
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=1000,
            C=1.0
        )
    else:
        raise ValueError("No such base: {}".format(base))

    # construct AdaBoost
    if algorithm == "SAMME":
        adaboost = SAMME(
            base_estimator=base_estimator,
            n_estimators=1000,
            n_classes=n_classes
        )
    elif algorithm == "SAMME.R":
        adaboost = SAMMER(
            base_estimator=base_estimator,
            n_estimators=1000,
            n_classes=n_classes
        )
    else:
        raise ValueError("No such algorithm: {}".format(algorithm))

    # train & sequential test
    adaboost.fit(tr_X, tr_Y)
    tr_errs = adaboost.report(tr_X, tr_Y)
    te_errs1 = adaboost.report(te_X1, te_Y1)
    te_errs2 = adaboost.report(te_X2, te_Y2)

    return tr_errs, te_errs1, te_errs2
