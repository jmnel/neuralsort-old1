# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:03:39 2020

@author: User
"""

from time import perf_counter

import os
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class NeuralSort(object):
    def __init__(self, method_name):
        self.batch_size = 1     # number of sequences
        self.seq_len = 3     # the length of a squence
        self.num_stitch = 4     # how many digit images are stitched together
        self.num_rows = self.num_stitch * 28   # nr of rows of a stitch image
        self.num_cols = 28    # nr of cols of a stitched image
        # the depth of an image (1 for gray scale images, 3 for RGB images)
        self.image_depth = 1

        self.tau = 5
        self.temperature = tf.convert_to_tensor(self.tau, dtype=tf.float32)
        self.initial_rate = 1e-4

        self.train_mode = True
        self.sess = tf.Session()
        self.saver = None

        self.method_name = method_name
        self.checkpoint_path = './checkpoints/sort-%s-b%d-seq%d-s%d-tau%.1f/best_model' %\
            (self.method_name, self.batch_size,
             self.seq_len, self.num_stitch, self.tau)
        # os.mkdir(self.checkpoint_path)

    def set_train_mode(self, mode=True):
        self.train_mode = mode

    def save_model(self, epoch):
        assert(False)
        # self.saver.save(self.sess, self.checkpoint_path + 'checkpoint', global_step=epoch)
        self.saver.save(self.sess, self.checkpoint_path)

    def load_model(self):
        assert(False)
        # filename = tf.train.latest_checkpoint(self.checkpoint_path)
        # if filename == None:
        #     raise Exception("No model found.")
        # print("Loaded model %s." % filename)
        print("Loaded model %s." % self.checkpoint_path)
        self.saver.restore(self.sess, self.checkpoint_path)

    # the relaxed permutation matrix

    def _bl_matmul(self, A, B):
        return tf.einsum('mij,jk->mik', A, B)

    # s: M x n x 1, ie, batch_size x seq_len x 1
    # neuralsort(s): M x n x n, ie, batch_size x seq_len x seq_len
    def compute_permu_matrix(self, s, tau=1):
        A_s = s - tf.transpose(s, perm=[0, 2, 1])
        A_s = tf.abs(A_s)        # As_ij = |s_i - s_j|
        n = tf.shape(s)[1]
        one = tf.ones((n, 1), dtype=tf.float32)
        B = self._bl_matmul(A_s, one @ tf.transpose(one)
                            )        # B_:k = (A_s)(one)
        K = tf.range(n) + 1        # K_k = k
        C = self._bl_matmul(s, tf.expand_dims(
            tf.cast(n + 1 - 2 * K, dtype=tf.float32), 0))        # C_:k = (n + 1 - 2k)s
        # P_k: = (n + 1 - 2k)s - (A_s)(one)
        P = tf.transpose(C - B, perm=[0, 2, 1])
        # P_k: = softmax( ((n + 1 - 2k)s - (A_s)(one)) / tau )
        P = tf.nn.softmax(P / tau, -1)

        return P

    # the deep CNN model

    def _conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def _weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        with tf.name_scope("reg"):
            return tf.Variable(initial)

    def _bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # A deep CNN model for 'classifying' digit images.
    # @param X An input tensor with the dimensions (seq. index, stitched image index, rows, cols)
    #          where cols=28, rows = 28 * (nr of digit images)
    # @param num_stitch How many digit images are stitched together.
    # @return An array of scores wrt stitched images.
    def deepnn(self, X, num_stitch):
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.

        shape = X.shape
        with tf.name_scope('reshape'):
            x_image = tf.expand_dims(tf.reshape(
                X, [-1, shape[2], shape[3]]), -1)
        # x_image = tf.reshape(x, [-1, l * 28, 28, 1])  # ????

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            self.W_conv1 = self._weight_variable([5, 5, 1, 32])
            self.b_conv1 = self._bias_variable([32])
            self.h_conv1 = tf.nn.relu(self._conv2d(
                x_image, self.W_conv1) + self.b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            self.h_pool1 = self._max_pool_2x2(self.h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            self.W_conv2 = self._weight_variable([5, 5, 32, 64])
            self.b_conv2 = self._bias_variable([64])
            self.h_conv2 = tf.nn.relu(self._conv2d(
                self.h_pool1, self.W_conv2) + self.b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            self.h_pool2 = self._max_pool_2x2(self.h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 64 features.
        with tf.name_scope('fc1'):
            self.W_fc1 = self._weight_variable(
                [self.num_stitch * 7 * 7 * 64, 64])
            self.b_fc1 = self._bias_variable([64])

        h_pool2_flat = tf.reshape(
            self.h_pool2, [-1, self.num_stitch * 7 * 7 * 64])
        self.h_fc1 = tf.nn.relu(
            tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        with tf.name_scope('fc2'):
            self.W_fc2 = self._weight_variable([64, 1])
            self.b_fc2 = self._bias_variable([1])

            h_fc1_flat = tf.reshape(self.h_fc1, [-1, 64])
            self.h_fc2 = tf.matmul(h_fc1_flat, self.W_fc2) + self.b_fc2

        return self.h_fc2

    # evaluation of accuracies
    # Pi: M x n x n row-stochastic

    def _prop_any_correct(self, P1, P2):
        z1 = tf.argmax(P1, axis=-1)
        z2 = tf.argmax(P2, axis=-1)
        eq = tf.equal(z1, z2)
        eq = tf.cast(eq, dtype=tf.float32)
        correct = tf.reduce_mean(eq, axis=-1)
        return tf.reduce_mean(correct)

    # Pi: M x n x n row-stochastic
    def _prop_correct(self, P1, P2):
        z1 = tf.argmax(P1, axis=-1)
        z2 = tf.argmax(P2, axis=-1)
        eq = tf.equal(z1, z2)
        correct = tf.reduce_all(eq, axis=-1)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    # build a list of (start,end) index tuples for batch processing

    def build_batch_indices(self, tot_examples):
        num_iter = tot_examples // self.batch_size
        lst_idx = [(k * self.batch_size, (k + 1) * self.batch_size)
                   for k in range(num_iter)]
        n_left = tot_examples - num_iter * self.batch_size
        if n_left > 0:
            lst_idx.append((tot_examples - self.batch_size, tot_examples))
        return lst_idx


class DetermNeuralSort(NeuralSort):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.best_acc = 0  # the best accuracy achieved so far

    def __delete__(self):
        self.sess.close()
        self.sess = None

    def compile(self):
        with tf.device("/cpu:0"):
            # the construction of loss function, accuracies, etc.
            self.x_data = tf.placeholder(
                shape=[None, self.seq_len, self.num_rows, self.num_cols], dtype=tf.float32)
            self.true_scores = tf.placeholder(
                shape=[None, self.seq_len], dtype=tf.float32)
            # self.x_data = tf.placeholder(shape = [self.batch_size, self.seq_len, self.num_rows, self.num_cols], dtype = tf.float32)
            # self.true_scores = tf.placeholder(shape = [self.batch_size, self.seq_len], dtype = tf.float32)

            P_true = self.compute_permu_matrix(
                tf.expand_dims(self.true_scores, 2), 1e-10)
            scores = tf.reshape(self.deepnn(self.x_data, self.num_stitch), [
                                self.batch_size, self.seq_len, 1])

            scores2 = tf.Print(scores, [scores])
#            scores2 = tf.Print(scores, [scores])

            P_hat = self.compute_permu_matrix(scores2, self.temperature)

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=P_true, logits=tf.log(P_hat + 1e-20), dim=2)
            losses = tf.reduce_mean(losses, axis=-1)
            self.loss = tf.reduce_mean(losses)

            self.per_correct = self._prop_correct(P_true, P_hat)
            self.per_any_correct = self._prop_any_correct(P_true, P_hat)

            # the optimizer
            opt = tf.train.AdamOptimizer(self.initial_rate)
            self.train_step = opt.minimize(self.loss)

            # # try the gradients
            # self.grad_b_conv1 = tf.gradients(self.loss, [self.b_conv1])[0]

#        self.saver = tf.train.Saver()

    def _train(self, epoch, images, values):
        loss_train = []
        lst_idx = self.build_batch_indices(images.shape[0])
#        print(lst_idx)
        for (start, end) in lst_idx:
            x_in = images[start: end]
            y_out = values[start: end]

            var = tf.trainable_variables()
            print(var)
#            for v in var:
#                print(var)
            exit()

            _, l = self.sess.run([self.train_step, self.loss],
                                 feed_dict={self.x_data: x_in, self.true_scores: y_out})
            loss_train.append(l)
            # print(" ")
            # print("gradient (b_conv1): ")
            # print(grad_b_conv1_val)

        print('Average loss: %.3f' % (np.mean(loss_train)), end=" ")

    def _test(self, epoch, images, values):
        p_cs = []
        p_acs = []
        lst_idx = self.build_batch_indices(images.shape[0])
        for (start, end) in lst_idx:
            x_in = images[start:end]
            y_out = values[start:end]
            p_c, p_ac = self.sess.run([self.per_correct, self.per_any_correct],
                                      feed_dict={self.x_data: x_in, self.true_scores: y_out})
            p_cs.append(p_c)
            p_acs.append(p_ac)

        p_c = np.mean(p_cs)
        p_ac = np.mean(p_acs)

        print("per. all-corr %.4f,    per. any-cor %.4f" %
              (p_c, p_ac), end=" ")
        if self.train_mode:
            if p_ac > self.best_acc:
                self.best_acc = p_ac
                print("saving the model...", end=" ")
#                self.save_model(epoch)

    def fit(self, ds_dicts, num_epochs):
        images_train, values_train = ds_dicts['train']
        images_valid, values_valid = ds_dicts['valid']
        images_test, values_test = ds_dicts['test']

        print("Training...")
        self.set_train_mode(True)
        self.best_acc = 0
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            t_start = perf_counter()
            print("Epoch: %03d " % epoch, end=" ")
            self._train(epoch, images_train, values_train)
            self._test(epoch, images_valid, values_valid)
            print(" ")
            print('train took {:.1f} seconds.'.format(
                perf_counter() - t_start))

        print("Testing...")
        self.set_train_mode(False)
        self.load_model()
        self._test(epoch, images_test, values_test)
        print(" ")


class StochNeuralSort(DetermNeuralSort):
    def __init__(self, method_name):
        super().__init__(method_name)

        self.repeat_times = 5   # the nr of times to repeat a batch

    def _sample_gumbel(self, shape, eps=1e-20):
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def compile(self):
        with tf.device("/gpu:0"):
            # the construction of loss function, accuracies, etc.
            self.x_data = tf.placeholder(
                shape=[None, self.seq_len, self.num_rows, self.num_cols], dtype=tf.float32)
            self.true_scores = tf.placeholder(
                shape=[None, self.seq_len], dtype=tf.float32)

            scores = tf.reshape(self.deepnn(self.x_data, self.num_stitch), [
                                self.batch_size, self.seq_len, 1])
            scores = tf.tile(scores, [self.repeat_times, 1, 1]) + self._sample_gumbel(
                [self.batch_size * self.repeat_times, self.seq_len, 1])

            scores2 = tf.Print(scores, [scores])

            P_hat = self.compute_permu_matrix(scores2, self.temperature)

            P_true = self.compute_permu_matrix(
                tf.expand_dims(self.true_scores, 2), 1e-10)
            P_true = tf.tile(P_true, [self.repeat_times, 1, 1])

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=P_true, logits=tf.log(P_hat + 1e-20), dim=2)
            losses = tf.reduce_mean(losses, axis=-1)
            self.loss = tf.reduce_mean(losses)

            self.per_correct = self._prop_correct(P_true, P_hat)
            self.per_any_correct = self._prop_any_correct(P_true, P_hat)

            # the optimizer
            opt = tf.train.AdamOptimizer(self.initial_rate)
            self.train_step = opt.minimize(self.loss)

#        self.saver = tf.train.Saver()
