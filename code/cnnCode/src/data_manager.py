from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from numpy import arange
from PIL import Image

class DataManager() :
  
  def __init__(self, batch_size=64, seed=0) :
    mnist = fetch_mldata('MNIST original')
    num_classes = 10
    
    self.batch_size = batch_size
    
    self.one_hot_encoder = LabelBinarizer()
    self.one_hot_encoder.fit(range(num_classes))
    
    np.random.seed(seed)
    N = len(mnist.data)
    p = np.random.permutation(N)
    
    x = list(map(lambda i : mnist.data[i], p))
    y = list(map(lambda i : int(mnist.target[i]), p))
    y = self.one_hot_encoder.transform(y)
    
    self.n_train = 50000
    self.n_val   = 10000
    self.n_test  = 10000
    
    n_train_val = self.n_train + self.n_val
    
    self.x_train, self.x_val, self.x_test = np.array(x[0:self.n_train]), np.array(x[self.n_train: n_train_val]), np.array(x[n_train_val: N])
    
    self.y_train, self.y_val, self.y_test = np.array(y[0:self.n_train]), np.array(y[self.n_train: n_train_val]), np.array(y[n_train_val: N])
  
  def batch(self, shuffle=True) :
    if shuffle :
      index = np.random.randint(self.n_train, size=self.batch_size)
      return self.x_train[index], self.y_train[index]
    
    else :
      x = self.x_train[0:self.batch_size]
      y = self.y_train[0:self.batch_size]
      
      self.x_train = np.vstack((self.x_train[self.batch_size:], x))
      self.y_train = np.vstack((self.y_train[self.batch_size:], y))
      return x, y
  
  def test_data(self) :
    return self.x_test, self.y_test
    
  def val_data(self) :
    return self.x_val, self.y_val
    
  def display(self, x, file=None) :
    img = x.reshape(28,28).astype(np.uint8)
    disp = Image.fromarray(img)
    if file is not None :
      disp.save(file)
    else :  
      disp.show()
