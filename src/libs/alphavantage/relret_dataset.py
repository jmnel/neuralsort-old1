import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

from random import randint

import torch
from torch.utils.data import DataLoader
import numpy as np
from pprint import pprint

from db_connectors import SQLite3Connector


class RelativeReturnsDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_path: Path,
                 size,
                 prediction_window=20,
                 sequence_len=5,
                 train=True,
                 transform=None):

        super().__init__()

        self.transform = transform

        db = SQLite3Connector.connect(data_path / 'clean.db')

        table = 'relative_returns_clean'
        schema = db.get_schema(table)

        symbols = [s['name'] for s in schema[1:]][0:sequence_len]

        data = db.select(table, symbols)
        db.close()

        offsets = [randint(0, len(data) - prediction_window - 1)
                   for _ in range(size)]

        for o in offsets:
            assert(o + prediction_window < len(data))

        self.data = list()
        for i in range(size):
            o = offsets[i]
            seq = torch.FloatTensor(data[o:o + prediction_window]).t()
            labels = torch.FloatTensor(
                data[o + prediction_window]).reshape((5, 1))
            self.data.append((seq, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.data[idx][0]
        label = self.data[idx][1]

        if self.transform:
            seq = self.transform(seq)

        return (seq, label)
