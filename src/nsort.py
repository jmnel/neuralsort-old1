import torch
import numpy as np
import math
from torchvision import datasets, transforms
from mnist_stitched_dataset import StitchedMNIST
import matplotlib.pyplot as plt

train_loader = torch.utils.data.DataLoader(
    StitchedMNIST('../data/mnist_stitched.pkl',
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[-0.0113, ], std=[0.9763, ])
                  ])),
    batch_size=100, shuffle=True, pin_memory=True)

data, label = next(iter(train_loader))


class CnnModel(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(
