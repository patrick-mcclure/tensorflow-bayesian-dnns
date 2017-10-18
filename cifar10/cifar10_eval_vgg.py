# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import bayesian_dropout

import cifar10_vgg as cifar10

parser = cifar10.parser


def eval_once(saver, summary_writer, mc, new_images, new_labels, images, labels, probabilities, top_k_op, mc_top_k_op, mc_probs, mc_loss, loss, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      mc_true_count = 0
      avg_nll = 0
      mc_avg_nll = 0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        x, y = sess.run([new_images, new_labels], feed_dict = {images:np.zeros(dtype=np.float32,shape=[FLAGS.batch_size,24,24,3]), mc:False, labels:np.zeros(dtype=np.int32,shape=[FLAGS.batch_size])})

        predictions, nll = sess.run([top_k_op,loss], feed_dict = {images: x, labels:y, mc:False})
        
        for n in range(10):
          if n == 0:
            p = sess.run(probabilities, feed_dict = {images: x, labels:y, mc:True, mc_probs: np.zeros(dtype=np.float32,shape=[FLAGS.batch_size,10])})
          else:
            p = p + sess.run(probabilities, feed_dict = {images: x, labels:y, mc:True, mc_probs: np.zeros(dtype=np.float32,shape=[FLAGS.batch_size,10])})
        p = p / 10
        mc_predictions, mc_nll = sess.run([mc_top_k_op, mc_loss], feed_dict = {images: x, labels:y, mc:True, mc_probs: p})

        true_count += np.sum(predictions)
        mc_true_count += np.sum(mc_predictions)
        avg_nll += nll
        mc_avg_nll += mc_nll
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      mc_precision = mc_true_count / total_sample_count
      avg_nll = avg_nll / total_sample_count
      mc_avg_nll = mc_avg_nll / total_sample_count
      print('%s: Accuracy = %.3f' % (datetime.now(), precision))
      print('%s: Loss = %.3f' % (datetime.now(), avg_nll))
      print('%s: MC Accuracy = %.3f' % (datetime.now(), mc_precision))
      print('%s: MC Cross Entropy = %.3f' % (datetime.now(), mc_avg_nll))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op,feed_dict = {images:np.zeros(dtype=np.float32,shape=[FLAGS.batch_size,24,24,3]), labels:np.zeros(dtype=np.int32,shape=[FLAGS.batch_size]), mc:True}))
      summary.value.add(tag='Accuracy', simple_value=precision)
      summary.value.add(tag='Loss', simple_value=avg_nll)
      summary.value.add(tag='MC Accuracy', simple_value=mc_precision)
      summary.value.add(tag='MC Loss', simple_value=mc_avg_nll)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    new_images, new_labels = cifar10.inputs(eval_data=eval_data)
    mc = tf.placeholder(tf.bool)
    images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 24, 24, 3])
    labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    mc_probs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 10])
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images,mc)
    probabilities = tf.nn.softmax(logits)
    loss = cifar10.loss(logits, labels)
    mc_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(labels,10) * tf.log(mc_probs), reduction_indices=[1]))

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    mc_top_k_op = tf.nn.in_top_k(mc_probs, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, mc, new_images, new_labels, images, labels, probabilities, top_k_op, mc_top_k_op, mc_probs, mc_loss, loss, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
