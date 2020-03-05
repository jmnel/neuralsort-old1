import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent / 'libs'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

from db_connectors import SQLite3Connector
from alphavantage import RelativeReturnsDataset
from lstm_model import LstmModel

torch.manual_seed(0)
device = torch.device('cpu')

num_layers = 4

train_size = 640
test_size = 160
train_batch_size = 1
test_batch_size = 1

epochs = 10
forecast_window = 100
num_seqences = 5


def _bl_matmul(mat_a, mat_b):
    return torch.einsum('mij,jk->mik', mat_a, mat_b)


def compute_permu_matrix(s: torch.FloatTensor, tau=1):
    mat_as = s - s.permute(0, 2, 1)
    mat_as = torch.abs(mat_as)
    n = s.shape[1]
    one = torch.ones(n, 1)
    b = _bl_matmul(mat_as, one @ one.transpose(0, 1))
    k = torch.arange(n) + 1
    d = (n + 1 - 2 * k).float().detach().requires_grad_(True).unsqueeze(0)
    c = _bl_matmul(s, d)
    mat_p = (c - b).permute(0, 2, 1)
    mat_p = F.softmax(mat_p / tau, -1)

    return mat_p


def _prop_any_correct(p1, p2):
    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.eq(z1, z2).float()
    correct = torch.mean(eq, axis=-1)
    return torch.mean(correct)


def _prop_correct(p1, p2):
    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.eq(z1, z2)
    correct = torch.all(eq, axis=-1).float()
    return torch.sum(correct)


def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (seq, label) in enumerate(train_loader):
        optimizer.zero_grad()

        seq = seq.to(device)
        label = label.to(device)
#        print(seq.shape)

#        plt.plot(seq[0, :, :].t(), linewidth=0.2)
#        plt.show()
#        exit()

        model.hidden_cell = (torch.zeros(num_layers, 1, 100).to(device),
                             torch.zeros(num_layers, 1, 100).to(device))

        scores = torch.zeros(num_seqences)

        for i in range(5):
            s = model(seq[0, i, :])
            scores[i] = s

        scores = scores.reshape((train_batch_size, num_seqences, 1))
        true_scores = label

        p_true = compute_permu_matrix(true_scores, 1e-10)
        p_hat = compute_permu_matrix(scores, 5)

        loss = -torch.sum(p_true * torch.log(p_hat + 1e-20), dim=-1).mean()

        loss.backward()
        optimizer.step()

        print(loss)

    print(f'epoch {epoch}')


data_path = Path(__file__).absolute().parents[2] / 'data'

train_loader = torch.utils.data.DataLoader(
    RelativeReturnsDataset(data_path, train_size, forecast_window, 5, True),
    batch_size=train_batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    RelativeReturnsDataset(data_path, test_size, forecast_window, 5, True),
    batch_size=test_batch_size, shuffle=True)

model = LstmModel(num_layers=num_layers)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):

    train(model, device, train_loader, optimizer, epoch)
