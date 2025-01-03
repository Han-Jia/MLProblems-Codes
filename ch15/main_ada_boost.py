import numpy as np

from overfit_ada_boost import train_ada_boost

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

if __name__ == "__main__":
    scene_choices = [False, True]
    datasets = ["BREASTCANCER", "DIGITS", "USPS"]
    algorithms = ["SAMME", "SAMME.R"]
    D = len(datasets)
    A = len(algorithms)

    colors = ["red", "green", "blue"]
    legends = ["训练集误差", "测试集1误差", "测试集2误差"]

    for domain_shift in scene_choices:
        fig = plt.figure(figsize=(12, 6))
        for d, dataset in enumerate(datasets):
            for a, algorithm in enumerate(algorithms):
                # 训练AdaBoost
                tr_errs, te_errs1, te_errs2 = train_ada_boost(
                    dataset=dataset, domain_shift=domain_shift,
                    base="DTC", algorithm=algorithm
                )

                xs = list(range(len(tr_errs)))

                plt.subplot(A, D, a * D + d + 1)
                ax = plt.gca()
                ax.plot(xs, tr_errs, color=colors[0], linestyle="solid")
                ax.plot(xs, te_errs1, color=colors[1], linestyle="dashed")
                ax.plot(xs, te_errs2, color=colors[2], linestyle="dotted")
                ax.legend(legends, fontsize=12)

                if a == 0:
                    ax.set_title(dataset, fontsize=16)

                if a == 1:
                    ax.set_xlabel("基学习器数目", fontsize=16)

                ax.set_ylabel(algorithm, fontsize=16)

                xticks = np.linspace(0, len(xs), 6).astype(np.int32)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(x) for x in xticks], fontsize=12)

        fig.tight_layout()
        fig.savefig(
            "ch14_AdaBoost-{}.jpg".format(domain_shift),
            dpi=300, bbox_inches=None
        )
        plt.show()
        plt.close()
