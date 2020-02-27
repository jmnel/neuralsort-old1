import torch
import torch.nn.functional as F

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

p_hat = tf.constant([[[0.1439, 0.0839, 0.7722],
                      [0.4946, 0.4132, 0.0922],
                      [0.4538, 0.5433, 0.0029]]])

p_true = tf.constant([[[0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0]]])
losses = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=p_true,
    logits=tf.log(p_hat + 1e-20), dim=2)

loss = tf.reduce_mean(losses)

sess = tf.Session()

res = sess.run(loss)

print(res)

p_true = torch.tensor([[[0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0]]])

p_hat = torch.tensor([[[0.1439, 0.0839, 0.7722],
                       [0.4946, 0.4132, 0.0922],
                       [0.4538, 0.5433, 0.0029]]])

logits = torch.log(p_hat)

# print(logits)


loss = -torch.sum(p_true * torch.log(p_hat + 1e-20), dim=2).mean()

print(loss)

# print(losses)
