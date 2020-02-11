from random import randint

import os.path as path
import torch
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt


class MnistSequenceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 num_stitched,
                 seq_length,
                 train_size,
                 test_size,
                 transform=None):
        super().__init__()

        self.transform = transform

        assert(num_stitched > 0)
        assert(seq_length > 1)

        data_path = path.dirname(path.realpath(__file__))
        data_path = path.join(data_path, '..', 'data')

        save_path = path.join(data_path, 'mnist_sequence.pkl')

        file_exists = path.exists(save_path)

        print(f'file exists={file_exists}')
        print(f'path={data_path}')

        train_load_data = datasets.MNIST(data_path, train=True, download=True,
                                         transform=transforms.ToTensor())
        test_load_data = datasets.MNIST(data_path, train=False, download=True,
                                        transform=transforms.ToTensor())

        train_load_batch_size = len(train_load_data)
        test_load_batch_size = len(test_load_data)

        train_load_dataloader = DataLoader(train_load_data,
                                           batch_size=train_load_batch_size,
                                           shuffle=True)
        test_load_dataloader = DataLoader(test_load_data,
                                          batch_size=train_load_batch_size,
                                          shuffle=True)

        train_digits_img, train_digits_labels = next(
            iter(train_load_dataloader))
        test_digits_img, test_digits_labels = next(iter(test_load_dataloader))

        train_digits_img = torch.reshape(
            train_digits_img, (train_load_batch_size, 28, 28))
        test_digits_img = torch.reshape(
            test_digits_img, (test_load_batch_size, 28, 28))

        def gen_stitch():
            picks = tuple(randint(0, train_load_batch_size - 1)
                          for k in range(num_stitched))

            stitch_img = torch.cat([
                train_digits_img[p] for p in picks], 1)

            stitch_label = sum(
                train_digits_labels[picks[k]] * 10**(num_stitched - k - 1)
                for k in range(num_stitched))

            return (stitch_img, stitch_label)

#        for i in range(test_size):

        def gen_sequence():
            sequence = tuple(gen_stitch() for j in range(seq_length))
            sequence_img = torch.cat([s[0] for s in sequence], 0)
            sequence_labels = tuple(s[1] for s in sequence)

            perm_sort = np.argsort(sequence_labels)

            sequence_labels = (sequence[p][1] for p in perm_sort)
            sequence_img = torch.cat([sequence[p][0] for p in perm_sort], 0)

            return (sequence_img, perm_sort)

#            print(perm_sort)
#            print(tuple(sequence_labels))
#            plt.imshow(sequence_img)
#            plt.show()
#            exit()

        x, y = gen_sequence()

        print(y)

        plt.imshow(x)
        plt.show()

        print('done')


foo = MnistSequenceDataset(num_stitched=4, seq_length=5,
                           train_size=10000, test_size=10000)
