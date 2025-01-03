import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Data import load_data
from utils import Averager, count_acc


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, X):
        logits = self.classifier(X)
        return logits


class DemoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = copy.deepcopy(X)
        self.Y = copy.deepcopy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.FloatTensor(self.X[i])
        y = torch.LongTensor([self.Y[i]])[0]
        return x, y


def train(model, train_loader, optimizer):
    model.train()

    avg_loss = Averager()
    avg_acc = Averager()

    # train 5 epochs
    for _ in range(5):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            # forward
            logits = model(batch_x)

            # calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss.add(loss.item())
            avg_acc.add(count_acc(logits, batch_y))

    loss = avg_loss.item()
    acc = avg_acc.item()

    return loss, acc


def test(model, loader):
    model.eval()

    avg_loss = Averager()
    avg_acc = Averager()

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            # only need forward
            logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)

            avg_loss.add(loss.item())
            avg_acc.add(count_acc(logits, batch_y))

    loss = avg_loss.item()
    acc = avg_acc.item()

    return loss, acc


def train_MLP(dataset, domain_shift, epochs=20):
    # hyper-parameters
    hidden_size = 1024
    batch_size = 64
    lr = 0.1

    # load data
    tr_X, tr_Y, te_X1, te_Y1, te_X2, te_Y2 = load_data(
        dataset=dataset,
        domain_shift=domain_shift
    )
    input_size = tr_X.shape[1]
    n_classes = len(np.unique(tr_Y))

    # construct dataset & dataloaders
    tr_set = DemoDataset(tr_X, tr_Y)
    te_set1 = DemoDataset(te_X1, te_Y1)
    te_set2 = DemoDataset(te_X2, te_Y2)

    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
    te_loader1 = DataLoader(te_set1, batch_size=batch_size, shuffle=False)
    te_loader2 = DataLoader(te_set2, batch_size=batch_size, shuffle=False)

    # construct model
    model = MLPClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        n_classes=n_classes
    )

    model.cuda()

    # construct optimizer & cosine scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    tr_errs, te_errs1, te_errs2 = [], [], []
    for ep in tqdm(range(epochs)):
        tr_loss, tr_acc = train(model, tr_loader, optimizer)
        te_loss1, te_acc1 = test(model, te_loader1)
        te_loss2, te_acc2 = test(model, te_loader2)

        lr_scheduler.step()

        tr_errs.append(1.0 - tr_acc)
        te_errs1.append(1.0 - te_acc1)
        te_errs2.append(1.0 - te_acc2)

    return tr_errs, te_errs1, te_errs2
