import torch
from scipy.signal import savgol_filter


def moving_averge(ys):
    n = len(ys)
    n1 = 2 * int(0.1 * n) - 1

    ys = savgol_filter(ys, n1, 3)
    return ys


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()
