import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt

from Data import load_data

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

for dataset in ["DIGITS", "USPS"]:
    for domain_shift in [False, True]:
        tr_X, tr_Y, te_X1, te_Y1, te_X2, te_Y2 = load_data(
            dataset=dataset,
            domain_shift=domain_shift
        )

        names = ["训练集", "测试集1", "测试集2"]

        fig = plt.figure(figsize=(6, 3))
        for d, X in enumerate([tr_X[0], te_X1[0], te_X2[0]]):
            w = int(np.sqrt(len(X)))
            img = X.reshape(w, w)

            plt.subplot(1, 3, d + 1)
            ax = plt.gca()
            ax.imshow(img)

            ax.set_title(names[d], fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(
            "ch14_Data-{}-{}.jpg".format(dataset, domain_shift),
            dpi=300, bbox_inches=None
        )
        plt.show()
        plt.close()
