import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

matplotlib.use("pgf")
pgf_config = {
    "font.family": 'serif',
    "font.size": 20,
    "pgf.rcfonts": False,
    "text.usetex": True,
    "pgf.preamble": [
        r"\usepackage{unicode-math}",
        # r"\setmathfont{XITS Math}",
        # 这里注释掉了公式的XITS字体，可以自行修改
        r"\setmainfont{Times New Roman}",
        r"\usepackage{xeCJK}",
        r"\xeCJKsetup{CJKmath=true}",
        r"\setCJKmainfont{SimSun}",
    ],
}
rcParams.update(pgf_config)

matplotlib.rc('axes', linewidth=2)


def pr_curve(dataset, x):
    tp, fp, fn = 0, 0, 0
    for value in dataset.values:
        if value[0] == 1 and value[1] >= x:  # 真正例
            tp += 1
        elif value[0] == 1 and value[1] < x:  # 假反例
            fn += 1
        elif value[0] == 0 and value[1] >= x:  # 假正例
            fp += 1
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return p, r


if __name__ == "__main__":
    data = pd.read_csv('data.csv', index_col='Index')
    data = data.sort_values(by='output', ascending=False)
    P_lst, R_lst = [], []
    x_lst = []
    for y in np.linspace(0, 0.999, 1000):
        P, R = pr_curve(data, y)
        P_lst.append(P)
        R_lst.append(R)
        x_lst.append(y)
    plt.plot(R_lst, P_lst, linewidth=2)
    plt.xlabel('R')
    plt.ylabel('P')
    # plt.title('P-R 曲线')
    plt.savefig('ch2_PR.pdf', bbox_inches='tight')
