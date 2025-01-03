import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from ada_boost import SAMME, SAMMER
from Data import load_data

from matplotlib import rcParams
from matplotlib import pyplot as plt

plt.style.use("ggplot")

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def compare_ada_boost(dataset, algorithms, n_estimators):
    # load data
    tr_X, tr_Y, te_X1, te_Y1, te_X2, te_Y2 = load_data(
        dataset=dataset,
        domain_shift=False
    )
    n_classes = len(np.unique(tr_Y))

    te_X = np.concatenate([te_X1, te_X2], axis=0)
    te_Y = np.concatenate([te_Y1, te_Y2], axis=0)

    # basic estimator
    base_estimator = DecisionTreeClassifier(
        max_depth=1
    )

    tr_accs, te_accs = [], []
    for algorithm in algorithms:
        for n_estimator in n_estimators:
            # train & test reproduced AdaBoost
            if algorithm == "SAMME":
                adaboost = SAMME(
                    base_estimator=base_estimator,
                    n_estimators=n_estimator,
                    n_classes=n_classes
                )
            elif algorithm == "SAMME.R":
                adaboost = SAMMER(
                    base_estimator=base_estimator,
                    n_estimators=n_estimator,
                    n_classes=n_classes
                )

            adaboost.fit(tr_X, tr_Y)
            tr_acc = np.mean(adaboost.predict(tr_X) == tr_Y)
            te_acc = np.mean(adaboost.predict(te_X) == te_Y)

            # sklearn adaboost
            adaboost_skl = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimator,
                learning_rate=1.0,
                algorithm=algorithm
            )
            adaboost_skl.fit(tr_X, tr_Y)
            tr_acc_skl = np.mean(adaboost_skl.predict(tr_X) == tr_Y)
            te_acc_skl = np.mean(adaboost_skl.predict(te_X) == te_Y)

            tr_accs.append([tr_acc, tr_acc_skl])
            te_accs.append([te_acc, te_acc_skl])

            print(dataset, algorithm, n_estimator, te_acc, te_acc_skl)
    return tr_accs, te_accs


def plot(datasets, algorithms, n_estimators, tr_accs, te_accs):
    D = len(datasets)
    A = len(algorithms)
    E = len(n_estimators)

    tr_accs = np.array(tr_accs).reshape((D, A, E, 2))
    te_accs = np.array(te_accs).reshape((D, A, E, 2))

    fig = plt.figure(figsize=(12, 6))

    for a in range(A):
        for d in range(D):
            plt.subplot(A, D, a * D + d + 1)
            ax = plt.gca()

            ys = te_accs[d, a, :, 0]
            ys_skl = te_accs[d, a, :, 1]
            xs = list(range(E))

            print(ys)
            print(ys_skl)

            ax.plot(xs, ys, marker="x", color="blue")
            ax.plot(xs, ys_skl, marker="+", color="red")

            if a == 0:
                ax.set_title(datasets[d], fontsize=18)

            if a == A - 1:
                ax.set_xlabel("基学习器数目", fontsize=16)

            if d == 0:
                ax.set_ylabel(algorithms[a], fontsize=16)

            ax.legend(["本章实现", "sklearn"], fontsize=14)

            ax.set_xticks(xs)
            ax.set_xticklabels([str(ne) for ne in n_estimators], fontsize=12)

    fig.tight_layout()
    fig.savefig(
        "ch14_compare-adaboost.jpg",
        dpi=300, bbox_inches=None
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    datasets = ["BREASTCANCER", "DIGITS", "USPS"]
    algorithms = ["SAMME", "SAMME.R"]
    n_estimators = [20, 100, 200, 500, 1000]

    tr_accs, te_accs = [], []
    for dataset in datasets:
        tr_accs_dset, te_accs_dset = compare_ada_boost(
            dataset, algorithms, n_estimators
        )
        tr_accs.append(tr_accs_dset)
        te_accs.append(te_accs_dset)

    plot(datasets, algorithms, n_estimators, tr_accs, te_accs)
