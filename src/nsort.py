import torch
import numpy as np
import math
from torchvision import datasets, transforms
from mnist_stitched_dataset import StitchedMNIST
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(41472, 4096)
#        self.fc2 = nn.Linear(10368, 5832)
        self.fc2 = nn.Linear(4096, 9999)
#        self.fc2 = nn.Linear( , 5832

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    batch_size = args['batch_size']

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
#        print(f'target=\n{target}')
#        loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % (2) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target).item()
#            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    torch.manual_seed(0)
    device = torch.device('cuda')

    BATCH_SIZE = 20
    TEST_BATCH_SIZE = 10
    EPOCHS = 3

    args = {'batch_size': BATCH_SIZE,
            'test_batch_size': TEST_BATCH_SIZE}

    train_loader = torch.utils.data.DataLoader(
        StitchedMNIST('../data/mnist_stitched.pkl',
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(
                              mean=(0.1307, ), std=(0.3081, ))
                      ])),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        StitchedMNIST('../data/mnist_stitched.pkl',
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(
                              mean=(0.1307, ), std=(0.3081, ))
                      ])),
        batch_size=TEST_BATCH_SIZE, shuffle=True, pin_memory=True)

    model = CnnModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, EPOCHS + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        optimizer.step()


if __name__ == '__main__':
    main()
