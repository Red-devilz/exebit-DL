from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os

from model import Model
from data_manager import DataManager

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'train_dir/',
                       """Directory where to write event logs """
                       """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                        """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('train', True,
                        """"Run the training procedure""")

def train(sess, model, dm):
  val_x, val_y = dm.val_data()

  sess.run(tf.global_variables_initializer())

  summary_log = tf.summary.FileWriter(os.path.join(FLAGS.train_dir,'summary'), sess.graph)

  for itr in range(1, FLAGS.max_steps+1):
    batch_x, batch_y = dm.batch()
    summary_log.add_summary(model.get_summary(batch_x, batch_y), itr)

    if itr % 200 == 0:
      val_accuracy = model.get_accuracy(val_x, val_y)
      print ('step %d : validation accuracy = %g' % (itr, val_accuracy))

    model.update(batch_x, batch_y)

  save_path = saver.save(sess, os.path.join(FLAGS.train_dir, 'chkpt/model.ckpt'))
  print("Model saved in file: %s" % save_path)
  summary_log.close()


def main(_):

  dm = DataManager()

  test_x, test_y = dm.test_data()

  with tf.Session() as sess:
    
    model = Model(sess)

    saver = tf.train.Saver()

    if FLAGS.train:
      train(sess, model, dm)

    else:
      saver.restore(sess, os.path.join(FLAGS.train_dir, 'chkpt/model.ckpt'))

    print ()
    print ('Test Accuracy = ', model.get_accuracy(test_x, test_y))

if __name__ == '__main__':
  tf.app.run()