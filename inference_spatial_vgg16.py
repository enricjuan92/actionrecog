import os, sys, time, glob, math, cv2
import urllib2 as urllib

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from video_processing import io
from datasets import imagenet
from models.vgg import vgg
from preprocessing import vgg_preprocessing
from datasets import ucf101_utils

slim = tf.contrib.slim

# PATHs definitions
SPATIAL_CKPT = 'checkpoints/spatial_vgg16.ckpt'

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Graph().as_default():

    images, labels = ucf101_utils.load_train_dataset(batch_size=5000, is_training=False)

    ph_images = tf.placeholder(tf.float32, [None, 224, 224, 3])

    # Create the model, fuse the default arg scope to configure the batch norm parameters
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(ph_images, num_classes=101, is_training=False, scope='spatial_vgg16')
        inference = tf.nn.softmax(logits)

    # Add ops to restore all the variables.
    init_op = tf.contrib.framework.assign_from_checkpoint_fn(model_path='checkpoints/finetuning_spatial_vgg16_2.ckpt',
                                                             var_list=slim.get_model_variables(),
                                                             ignore_missing_vars=True)
    num_steps = 100
    batch_size = 50

    with tf.Session() as sess:  # set up the session

        init_op(sess)

        for step in range(num_steps):

            offset = (step * batch_size) % (labels.shape[0] - batch_size)
            start = offset
            end = (offset + batch_size)

            # Generate a minibatch.
            batch_data = images[start:end, :]
            batch_labels = labels[start:end, :]

            feed_dict = {ph_images: batch_data}

            predictions = sess.run(inference, feed_dict=feed_dict)

            print predictions.shape, np.amax(predictions) * 100

            if (step % 5 == 0):
                print('Step: %d From: %d To: %d' % (step, start, end))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))


