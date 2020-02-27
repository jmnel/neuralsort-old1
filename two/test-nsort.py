# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:12:37 2020

@author: User
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.framework import ops
from nsort import DetermNeuralSort, StochNeuralSort
from MNISTDatasetV2 import MNISTDataset

#import torch
#import torch.nn.functional as F
#from mnist_sequence_dataset import MnistSequenceDataset


if __name__ == "__main__":
    # set random seeds
    tf.set_random_seed(94305)
    # tf.random.set_seed(94305)
    random.seed(94305)
    ops.reset_default_graph()

    # the model
    ns = DetermNeuralSort("deterministic")
    # ns = StochNeuralSort("Stochastic")

    ns.batch_size = 64    # number of sequences
    ns.seq_len = 3     # the length of a squence
    ns.num_stitch = 4     # how many digit images are stitched together
    ns.num_rows = ns.num_stitch * 28   # nr of rows of a stitch image
    ns.num_cols = 28    # nr of cols of a stitched image
    # the depth of an image (1 for gray scale images, 3 for RGB images)
    ns.image_depth = 1
    ns.tau = 5
    ns.temperature = tf.convert_to_tensor(ns.tau, dtype=tf.float32)
    ns.initial_rate = 1e-4

    # make the datasets （64:16:20）
    num_seq_train = 6400
    num_seq_valid = 1600
    num_seq_test = 2000
    ds = MNISTDataset()
    ds.load_data()
    images_train, values_train = ds.make_dataset(
        ds.train, ns.num_stitch, ns.seq_len, num_seq_train)
    images_valid, values_valid = ds.make_dataset(
        ds.train, ns.num_stitch, ns.seq_len, num_seq_valid)
    images_test, values_test = ds.make_dataset(
        ds.train, ns.num_stitch, ns.seq_len, num_seq_test)

    # TORCH STUFF
#    train_loader = torch.utils.data.DataLoader(
#        MnistSequenceDataset(
#            num_stitched=ns.num_stitch,
#            seq_length=ns.seq_len,
#            size=6400),
#        batch_size=ns.batch_size,
#        shuffle=False)
#    pt_images, pt_values = next(iter(train_loader))

#    print(f'tf: {images_train.shape}')
#    print(f'torch: {pt_images.shape}')

    # training the model
    ns.compile()
    ns.fit(ds_dicts={'train': (images_train, values_train),
                     'valid': (images_valid, values_valid),
                     'test': (images_test, values_test)},
           num_epochs=50)
    # del ns

    # clean up the default graph
    ops.reset_default_graph()
