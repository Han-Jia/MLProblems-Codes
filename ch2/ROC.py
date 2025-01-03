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


def roc_curve(dataset, x):
    tp, tn, fp, fn = 0, 0, 0, 0
    for value in dataset.values:
        if value[0] == 1 and value[1] >= x:  # 真正例
            tp += 1
        elif value[0] == 1 and value[1] < x:  # 假反例
            fn += 1
        elif value[0] == 0 and value[1] >= x:  # 假正例
            fp += 1
        elif value[0] == 0 and value[1] < x:
            tn += 1  # 真反例
    fpr = fp / (tn + fp)  # 假正例率
    tpr = tp / (tp + fn)  # 真正例率
    return fpr, tpr


if __name__ == "__main__":
    data = pd.read_csv('data.csv', index_col='Index')
    data = data.sort_values(by='output', ascending=False)
    FPR_lst, TPR_lst = [], []
    for y in np.linspace(0, 0.999, 1000):
        FPR, TPR = roc_curve(data, y)
        FPR_lst.append(FPR)
        TPR_lst.append(TPR)
    plt.plot(FPR_lst, TPR_lst, linewidth=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.title('TPR-FPR Curve')
    plt.savefig('ch2_ROC.pdf', bbox_inches='tight')

    l_rank, m1, m2 = 0, 0, 0
    for value1 in data.values:
        if value1[0] == 1:
            m1 += 1
            for value2 in data.values:
                if value2[0] == 0 and value1[1] < value2[1]:
                    l_rank += 1
                elif value2[0] == 0 and value1[1] == value2[1]:
                    l_rank += 0.5
        else:
            m2 += 1
    l_rank = l_rank / (m1 * m2)
    AUC = 1 - l_rank
    print(AUC)
