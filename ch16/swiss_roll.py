from sklearn.datasets import make_swiss_roll

# 生成瑞士卷流型，X代表流型上的点的三维坐标，y代表该点在流型上的本真"位置"
X, y = make_swiss_roll(1000, random_state=0)
