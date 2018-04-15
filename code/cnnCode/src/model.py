from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from tf_util import *


class Model:

  def __init__(self, sess, learning_rate=1e-3):
    self.sess = sess

    # Define Architecture

    self.x = tf.placeholder(tf.float32, shape=[None, 784], name='input_flat_image')
    self.y = tf.placeholder(tf.int32, shape=[None, 10], name='input_label')

    x_im = tf.reshape(self.x, [-1, 28, 28, 1])

    with tf.variable_scope('conv_pool_1'):
      conv3_32 = conv2d(x_im, 32, 3, padding='SAME', name='conv3_32')
      pool2x2 = max_pool(conv3_32, 2, 2, name='max_pool_2x2')

    with tf.variable_scope('conv_pool_2'):
      conv3_32 = conv2d(pool2x2, 32, 3, padding='SAME', name='conv3_32')
      pool2x2 = max_pool(conv3_32, 2, 2, name='max_pool_2x2')

    self.cnn_code = tf.reshape(pool2x2, [-1, 7*7*32])

    fc1 = fc(self.cnn_code, 500, name='fc1')
    
    self.y_un = fc(fc1, 10, activation=None, name='unnormalized_prediction')

    self.prob = tf.nn.softmax(self.y_un)


    # Define Loss Function

    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_un))
    tf.summary.scalar('cross_entropy', self.cross_entropy)


    self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

    correct_prediction = tf.equal(tf.argmax(self.y_un, 1), tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.merged = tf.summary.merge_all()

  def update(self, batch_x, batch_y):
    return self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})

  def get_accuracy(self, batch_x, batch_y):
    return self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})

  def get_loss(self, batch_x, batch_y):
    return self.sess.run(self.cross_entropy, feed_dict={self.x: batch_x, self.y: batch_y})    

  def get_summary(self, batch_x, batch_y):
    return self.sess.run(self.merged, feed_dict={self.x: batch_x, self.y: batch_y})

