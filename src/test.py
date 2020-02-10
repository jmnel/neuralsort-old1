import torch
import numpy as np
from torchvision import transforms

from mnist_stitched_dataset import StitchedMNIST

import torch.nn as nn
import torch.nn.functional as F


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(41472, 2592)
        self.fc2 = nn.Linear(2592, 648)
        self.fc3 = nn.Linear(648, 40)

    def forward(self, x):

        print(x.shape[0])

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
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        x = x.reshape((x.shape[0], 10, 4))
        x = F.log_softmax(x, dim=1)

        return x
#        o[:10] = F.log_softmax(x[:10], dim=1)
#        o[10:20]
#        output = F.log_softmax(x, dim=1)
        # output = torch.cat([
        #     F.log_softmax(x[:10], dim=1),
        #     F.log_softmax(x[10:20], dim=1),
        #     F.log_softmax(x[20:30], dim=1),
        #     F.log_softmax(x[30:40], dim=1)])
#        return output


BATCH_SIZE = 2
train_loader = torch.utils.data.DataLoader(
    StitchedMNIST('../data/mnist_stitched.pkl',
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(
                          mean=(0.1307, ), std=(0.3081, ))
                  ])),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

model = CnnModel()

x, target = next(iter(train_loader))
target = target.long()

print(f'x = {x.shape}')

o = model(x)

t = torch.zeros(2, 4).long()

print(f't = {t.shape}')
print(f'ta = {target.shape}')

loss = F.nll_loss(o, target)

print(f'o = {o.shape}')
#x = torch.zeros((4, 10)).reshape((1, 4, 10))
#target = torch.tensor([0, 1, 3, 9])

# (N, C, d1)
# (N, d1)
# (
#x[0, 0, 0] = 1
#x[0, 1, 1] = 1
#x[0, 2, 3] = 1
#x[0, 3, 8] = 1

#loss = F.nll_loss(x, target, reduce=True, reduction='sum')

# print(loss)

# x, y = next(iter(train_loader))

# o = model(x)

# y = y.view(BATCH_SIZE, 4)

# y = y.long()

# print('x={}'.format(x.shape))
# print('y={}'.format(y.shape))
# print('o={}'.format(o.shape))

# l0 = F.nll_loss(o[:, :10], y[:, 0])


# print('l0={}'.format(l0))
#
