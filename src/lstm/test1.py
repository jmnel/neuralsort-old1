import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

import sys

for s in sys.path:
    print(s)

input_dim = 5
hidden_dim = 10
n_layers = 1

lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

batch_size = 1
seq_len = 1

inp = torch.randn(batch_size, seq_len, input_dim)
hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
cell_state = torch.rand(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)

out, hidden = lstm_layer(inp, hidden)
print(out.shape)
print(hidden)
