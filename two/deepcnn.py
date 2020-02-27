import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCnn(nn.Module):
    def __init__(self, num_digits):
        super(DeepCnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 1, 1)
        self.fc1 = nn.Linear(12544, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        exit()

        return x
