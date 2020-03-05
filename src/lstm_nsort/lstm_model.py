import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmModel(nn.Module):

    def __init__(self,
                 num_layers=1,
                 hidden_size=100):

        super().__init__()

        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=num_layers)

        self.linear = nn.Linear(in_features=100,
                                out_features=1)
        self.hidden_cell = (torch.zeros(num_layers, 1, hidden_size),
                            torch.zeros(num_layers, 1, hidden_size))

    def forward(self, x):

        lstm_out, self.hidden_cell = self.lstm(x.view(len(x), 1, -1),
                                               self.hidden_cell)

        y = self.linear(lstm_out.view(len(x), -1))

        return y[-1]
