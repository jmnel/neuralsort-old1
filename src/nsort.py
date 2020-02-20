import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from deepcnn import DeepCnn
from mnist_sequence_dataset import MnistSequenceDataset
# from mnist_stitched_dataset import StitchedMNIST

torch.manual_seed(0)
device = torch.device('cpu')

BATCH_SIZE = 10
TEST_BATCH_SIZE = 1
EPOCHS = 3
NUM_DIGITS = 4
NUM_SEQUENCES = 5


def _bl_matmul(mat_a, mat_b):
    return torch.einsum('mij,jk->mik', mat_a, mat_b)


def compute_permu_matrix(s: torch.FloatTensor, tau=1):
    mat_as = s - s.permute(0, 2, 1)
    mat_as = torch.abs(mat_as)
    n = s.shape[1]
    one = torch.ones(n, 1)
    b = _bl_matmul(mat_as, one @ one.transpose(0, 1))
    k = torch.arange(n) + 1
#    d = torch.tensor(n + 1 - 2 * k).unsqueeze(0)
    d = (n + 1 - 2 * k).float().detach().requires_grad_(True).unsqueeze(0)
    c = _bl_matmul(s, d)
    mat_p = (c - b).permute(0, 2, 1)
    mat_p = F.softmax(mat_p / tau, -1)

    return mat_p


def _prop_any_correct(p1, p2):
    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.equal(z1, z2)
    eq = eq.type(torch.FloatTensor)
    correct = torch.mean(eq, axis=-1)
    return torch.mean(correct)

#
# s = torch.randn(2, 5, 1)

# foo = compute_permu_matrix(s)

# print(foo.shape)
# print(foo)


def train(model, device, train_loader, optimizer, epoch):

    #    print(f'Train Epoch: {epoch}')

    model.train()

    avg_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):

        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        batch_losses = torch.zeros(BATCH_SIZE)

        bp_true = torch.zeros(BATCH_SIZE, NUM_SEQUENCES, NUM_SEQUENCES)
        bp_hat = torch.zeros(BATCH_SIZE, NUM_SEQUENCES, NUM_SEQUENCES)

        scores = torch.zeros(BATCH_SIZE, NUM_SEQUENCES)

        for i in range(BATCH_SIZE):

            for j in range(NUM_SEQUENCES):
                s = model(x[i:i + 1, j:j + 1, :, :])
#                print(f's={s}')

                scores[i, j] = s

#        print('foo')
#        print(scores)

#        exit()

        scores = scores.reshape((BATCH_SIZE, NUM_SEQUENCES, 1))
        true_scores = y.reshape((BATCH_SIZE, NUM_SEQUENCES, 1))
        true_scores = true_scores.type(torch.FloatTensor)

        p_true = compute_permu_matrix(true_scores, 1e-10)
        p_hat = compute_permu_matrix(scores, 5)

#        foo = torch.log(p_hat + 1e-20, dim=1)
        foo = torch.log(p_hat + 1e-20)
#        foo = torch.log(foo)
        loss = -torch.sum(p_true * foo, dim=1).mean()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#            print(p_true[0])
#            print(p_hat[0])
#            print(scores[0])

#        bp_true[i] = p_true
#        bp_hat[i] = p_hat

#        scores = scores.reshape((1, 5, 1))
#        true_scores = seq_s_true.reshape((1, 5, 1))
#        print(true_scores.shape)
#        print(scores.shape)

#        true_scores = true_scores.type(torch.FloatTensor)

#        mat_p_true = compute_permu_matrix(true_scores, 1e-10)
#        mat_p_hat = compute_permu_matrix(scores, 5)

#        print(mat_p_true)
#        print(mat_p_hat)

#        logits = torch.log(mat_p_hat + 1e-20)

#        foo = F.log_softmax(mat_p_hat, dim=1)
#        loss = -torch.sum(mat_p_true * foo, dim=1).mean()

#        loss.backward()
#        optimizer.step()

#        print(loss)

#        exit()


def test(model, device, test_loader):
    pass


#device = torch.device('cpu')

train_loader = torch.utils.data.DataLoader(
    MnistSequenceDataset(
        num_stitched=4, seq_length=5,
        size=10000),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    MnistSequenceDataset(
        num_stitched=4, seq_length=5,
        train=False, size=10000),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# print(len(train_loader.dataset))

model = DeepCnn(num_digits=NUM_DIGITS)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
#    optimizer.step()
# data = next(iter(train_loader))
