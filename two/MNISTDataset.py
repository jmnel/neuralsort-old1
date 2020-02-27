# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 20:26:34 2020

@author: User
"""

import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data



## This class manages the MNIST datasets. It builds the datasets for the neuralsort
#  algorithm. Following are some terms used in this algorithm:
#       stitched image  --- an image formed by stacking several MNIST digit images
#       sequence --- a list of stitched images
#
class MNISTDataset:
    def __init__(self):
        super().__init__()


    ## Extract the images of same digits to an array, then put the arrays into a list.
    #  For example, images wrt digit 0 are put into one array which is the first element in the list.
    #  Each row of the array contains the data for one image, therefore the number of rows indicates
    #  how many images are stored in that array. The image size is 28 x 28.
    #
    # @param data The data of type tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet
    # @return A list of arrays of image data for each digit.
    def _extract_images(self, data):
        return [ data.images[np.nonzero(data.labels[:, d])] for d in range(10) ]


    ## Load the MNIST dataset through TensorFlow routine.
    def load_data(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.train = self._extract_images(mnist.train)
        self.valid = self._extract_images(mnist.validation)
        self.test  = self._extract_images(mnist.test)


    ## Generate a sequence of stitched images.
    # @param digit_images The list of arrays of digit images.
    # @param num_stitch The number of MNIST digit images stitched together.
    # @param seq_len The length of each sequence of stitched images.
    # @param low The minimum integer that the digits in the stitched image may form.
    # @param high The maximum integer that the digits in the stitched image may form.
    # @return An array of stitched images with shape (# of stitched image, rows=# of digits times 28, cols=28)
    def _make_a_sequence(self, digit_images, num_stitch, seq_len, low, high):
        seq = []  # the sequence of stitched images
        vals = []  # the number that is formed by the digits in the stitched image
        for i in range(seq_len):
            stitch = []
            num = random.randint(low, high)
            vals.append(num)
            for j in range(num_stitch):
                digit = num%10
                num //= 10
                ref = digit_images[digit]
                idx = np.random.randint(0, ref.shape[0])
                stitch.insert(0, ref[idx])

            seq.append( np.concatenate(stitch).reshape(-1, 28) )

        return np.stack(seq), np.array(vals)


    ## Generate sequences of stitched digit images for the neural sort algorithm.
    # @param digit_images The list of arrays of digit images.
    # @param num_stitch The number of MNIST digit images stitched together.
    # @param seq_len The length of each sequence of stitched images.
    # @param num_seq How many sequences of stitched images to make.
    # @return An array of sequences of stitched digit images.
    def make_dataset(self, digit_images, num_stitch, seq_len, num_seq):
        low, high = 0, 10**num_stitch - 1
        seq_images = np.zeros((num_seq, seq_len, num_stitch*28, 28))
        seq_values = np.zeros((num_seq, seq_len))
        for i in range(num_seq):
            seq_images[i,], seq_values[i,] = self._make_a_sequence(digit_images, num_stitch, seq_len, low, high)

        return seq_images, seq_values


    ## Generate datasets for training, validation, and testing puporse.
    # @return A list of tuples of arrays of images and values.
    def make_all_datasets(self, num_stitch, seq_len, num_seq_train, num_seq_val, num_seq_test):
        return [ self.make_dataset(self.train, num_stitch, seq_len, num_seq_train),\
                 self.make_dataset(self.valid, num_stitch, seq_len, num_seq_val),\
                 self.make_dataset(self.test , num_stitch, seq_len, num_seq_test) ]



