from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


def conv2d(inputs, nb_filters, kernel_size, strides=1, padding='valid', activation=tf.nn.relu, regularizer_scale=0.05, name=None):
  initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
  l2_regularizer = tf.contrib.layers.l2_regularizer(regularizer_scale)  

  return tf.layers.conv2d(inputs, nb_filters, kernel_size,
                          strides=strides,
                          padding=padding,
                          activation=activation,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2_regularizer,
                          name=name)

def max_pool(inputs, pool_size, strides, name=None):
  return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], 'VALID', name=name)

def fc(inputs, units, activation=tf.nn.relu, regularizer_scale=0.05, name=None):
  initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
  l2_regularizer = tf.contrib.layers.l2_regularizer(regularizer_scale)  

  return tf.layers.dense(inputs, units, 
                         activation=activation,
                         kernel_initializer=initializer,
                         kernel_regularizer=l2_regularizer,
                         name=name)