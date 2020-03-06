import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmModel(nn.Module):

    def __init__(self,
                 num_layers=1,
                 hidden_size=100):

        super().__init__()

        self.lstm = nn.LSTM(input_size=5,
                            hidden_size=hidden_size,
                            num_layers=1, batch_first=True)

        self.linear = nn.Linear(in_features=100,
                                out_features=5)
        self.hidden_cell = (torch.zeros(1, 1, 100),
                            torch.zeros(1, 1, 100))

    def forward(self, x):

        #        print(x.shape)
        #        x = x.reshape((1, 10, 5))

        #        print('here')
        #        print(x.shape)
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)

#        print(lstm_out.shape)

#        lstm_out, self.hidden_cell = self.lstm(x.view(len(x), 1, -1),
#                                               self.hidden_cell)

        y = self.linear(lstm_out)

#        print(y.shape)

        last = y[:, -1, :]

#        print(last.shape)
#        print(last)
#        exit()

        return y[:, -1, :]
