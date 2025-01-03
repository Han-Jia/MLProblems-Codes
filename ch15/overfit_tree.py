import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from Data import load_data


def train_tree(dataset, domain_shift, max_depths=None):
    if max_depths is None:
        max_depths = np.arange(32) + 1

    tr_X, tr_Y, te_X1, te_Y1, te_X2, te_Y2 = load_data(
        dataset=dataset,
        domain_shift=domain_shift
    )

    tr_errs, te_errs1, te_errs2 = [], [], []
    for max_depth in tqdm(max_depths):
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(tr_X, tr_Y)

        tr_errs.append(np.mean(clf.predict(tr_X) != tr_Y))
        te_errs1.append(np.mean(clf.predict(te_X1) != te_Y1))
        te_errs2.append(np.mean(clf.predict(te_X2) != te_Y2))

    return tr_errs, te_errs1, te_errs2
