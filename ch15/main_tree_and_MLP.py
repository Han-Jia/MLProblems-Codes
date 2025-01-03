import numpy as np

from overfit_tree import train_tree
from overfit_MLP import train_MLP

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
    D = len(datasets)

    colors = ["red", "green", "blue"]
    legends = ["训练集误差", "测试集1误差", "测试集2误差"]

    for domain_shift in scene_choices:
        fig = plt.figure(figsize=(12, 6))
        for d, dataset in enumerate(datasets):
            # 训练决策树
            tr_errs, te_errs1, te_errs2 = train_tree(dataset, domain_shift)

            xs = list(range(len(tr_errs)))

            plt.subplot(2, D, d + 1)
            ax = plt.gca()
            ax.plot(xs, tr_errs, color=colors[0], linestyle="solid")
            ax.plot(xs, te_errs1, color=colors[1], linestyle="dashed")
            ax.plot(xs, te_errs2, color=colors[2], linestyle="dotted")
            ax.legend(legends, fontsize=12)
            ax.set_title(dataset, fontsize=16)

            xticks = np.linspace(0, len(xs), 6).astype(np.int32)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks], fontsize=12)
            ax.set_xlabel("决策树最大深度", fontsize=16)

            # 训练多层感知机
            tr_errs, te_errs1, te_errs2 = train_MLP(
                dataset, domain_shift, epochs=200
            )
            xs = list(range(len(tr_errs)))

            plt.subplot(2, D, d + D + 1)
            ax = plt.gca()
            ax.plot(xs, tr_errs, color=colors[0], linestyle="solid")
            ax.plot(xs, te_errs1, color=colors[1], linestyle="dashed")
            ax.plot(xs, te_errs2, color=colors[2], linestyle="dotted")
            ax.legend(legends, fontsize=12)
            ax.set_xlabel("多层感知机训练轮数", fontsize=16)

            xticks = np.linspace(0, len(xs), 6).astype(np.int32)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks], fontsize=12)

        fig.tight_layout()
        fig.savefig(
            "ch14_Tree-MLP-{}.jpg".format(domain_shift),
            dpi=300, bbox_inches=None
        )
        plt.show()
        plt.close()
