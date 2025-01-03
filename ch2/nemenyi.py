import matplotlib.pyplot as plt

A, B, C, D, E = (3.8, 4, 3.2, 1.2, 2.8)  # 平均序值
CD = 2.728  # 经计算得到的CD值
limits = (5, 1)
fig, ax = plt.subplots(figsize=(5, 1.8))
plt.subplots_adjust(left=0.2, right=0.8)

ax.set_xlim(limits)
ax.set_ylim(0, 1)
ax.spines['top'].set_position(('axes', 0.6))
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_visible(False)
for pos in ["bottom", "left", "right"]:
    ax.spines[pos].set_visible(False)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
arrow_props = dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90")
kw = dict(xycoords='data', textcoords="axes fraction", arrowprops=arrow_props, va="center")

# 绘制每个方法的对应序值和名称
ax.annotate("A", xy=(A, 0.6), xytext=(0, 0.32), ha="left", **kw)
ax.annotate("B", xy=(B, 0.6), xytext=(0, 0.48), ha="left", **kw)
ax.annotate("C", xy=(C, 0.6), xytext=(0., 0.16), ha="left", **kw)
ax.annotate("D", xy=(D, 0.6), xytext=(1, 0.5), ha="right", **kw)
ax.annotate("E", xy=(E, 0.6), xytext=(1, 0.), ha="right", **kw)

ax.plot([D, D + CD], [0.55, 0.55], color="k", lw=3)  # 绘制长为CD的线段
plt.tight_layout()
plt.savefig('ch2_nemenyi.pdf', bbox_props='tight')
