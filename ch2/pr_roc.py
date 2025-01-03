import matplotlib
import matplotlib.pyplot as plt

# font = {'family' : 'Times New Roman',
#         'weight' : 'bold',
#         'size'   : 20}

# matplotlib.rc('font', **font)
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# # matplotlib.rc('axes', linewidth=2)
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

P_lst, R_lst = [0, 1, 1, 0.67, 0.75, 0.6, 0.67, 0.57, 0.5], [0, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]
plt.plot(R_lst, P_lst, linewidth=2)
plt.xlabel('R')
plt.ylabel('P')
# plt.title('P-R Curve')
plt.savefig('ch2_8inst-PR.pdf', bbox_inches='tight')

plt.clf()
FPR_lst, TPR_lst = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]
plt.plot(FPR_lst, TPR_lst, linewidth=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
# plt.title('P-R Curve')
plt.savefig('ch2_8inst-ROC.pdf', bbox_inches='tight')
