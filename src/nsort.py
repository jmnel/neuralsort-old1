import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from deepcnn import DeepCnn
from mnist_stitched_dataset import StitchedMNIST

torch.manual_seed(0)
device = torch.device('cpu')

BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
EPOCHS = 3
NUM_DIGITS = 4


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


s = torch.randn(2, 5, 1)

foo = compute_permu_matrix(s)

print(foo.shape)
print(foo)

# train_loader = torch.utils.data.DataLoader(
#    StitchedMNIST('../data/mnist_stitched.pkl',
#                  transform=transforms.Compose([
#                      transforms.ToTensor(),
#                      transforms.Normalize(
#                          mean=(0.1307, ), std=(0.3081, ))
#                  ])),
#    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(
#    StitchedMNIST('../data/mnist_stitched.pkl',
#                  transform=transforms.Compose([
#                      transforms.ToTensor(),
#                      transforms.Normalize(
#                          mean=(0.1307, ), std=(0.3081, ))
#                  ])),
#    batch_size=TEST_BATCH_SIZE, shuffle=True, pin_memory=True)

#model = DeepCnn(num_digits=NUM_DIGITS)

#x, y_truth = next(iter(train_loader))

#y = model(x)

# print(f'x={x.shape}')
# print(f'y_truth={y_truth.shape}')
# print(y_truth)
# print(f'y={y}')
