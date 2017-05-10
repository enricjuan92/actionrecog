import os, sys, time
import urllib2 as urllib

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from video_processing import io

from datasets import imagenet
from models.vgg import vgg
from preprocessing import vgg_preprocessing

slim = tf.contrib.slim

# PATHs definitions
VIDEOS_PATH = 'resources/'
SPATIAL_CKPT = 'checkpoints/spatial_vgg16.ckpt'
TEMPORAL_CKPT = 'checkpoints/temporal_vgg16.ckpt'


with tf.Graph().as_default():

    spatial_ph = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    temporal_ph = tf.placeholder(shape=[None, 224, 224, 20], dtype=tf.float32)

    image = imread('resources/frame000044.jpg', mode='RGB')
    image = imresize(image, (224, 224))
    rgb_image = tf.expand_dims(image, 0)

    gray_image = tf.image.rgb_to_grayscale(rgb_image)
    stack_image = tf.concat([gray_image, gray_image, gray_image, gray_image, gray_image,
                            gray_image, gray_image, gray_image, gray_image, gray_image,
                            gray_image, gray_image, gray_image, gray_image, gray_image,
                            gray_image, gray_image, gray_image, gray_image, gray_image], 3)

    spatial_batch = tf.cast(rgb_image, tf.float32)
    temporal_batch = tf.cast(stack_image, tf.float32)

    # Create the model, fuse the default arg scope to configure the batch norm parameters
    with slim.arg_scope(vgg.vgg_arg_scope()):
        spatial_logits, _ = vgg.vgg_16(spatial_batch, num_classes=101, is_training=False, scope='spatial_vgg16')
        temporal_logits, _ = vgg.vgg_16(temporal_batch, num_classes=101, is_training=False, scope='temporal_vgg16')

    spatial_probs = tf.nn.softmax(spatial_logits)
    temporal_probs = tf.nn.softmax(temporal_logits)

    # Define the loss functions and get the total loss.
    # loss = slim.losses.softmax_cross_entropy(logits, labels)
    # total_loss = slim.losses.get_total_loss()
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # create_train_op ensures that each time we ask for the loss, the update_ops
    # are run and the gradients being computed are applied too
    # train_op = slim.learning.create_train_op(total_loss, optimizer)
    # logdir = ...  # Where checkpoints are stored


    # Add ops to restore all the variables.
    spatial_init = tf.contrib.framework.assign_from_checkpoint_fn(model_path=SPATIAL_CKPT,
                                                                  var_list=slim.get_model_variables(),
                                                                  ignore_missing_vars=True)
    temporal_init = tf.contrib.framework.assign_from_checkpoint_fn(model_path=TEMPORAL_CKPT,
                                                                  var_list=slim.get_model_variables(),
                                                                  ignore_missing_vars=True)

    # slim.learning.train(train_op, log_dir, init_fn=init_fn)

    with tf.Session() as sess:  # set up the session

        spatial_init(sess)
        temporal_init(sess)

        # np_images, probabilities = sess.run(probabilities, feed_dict={x: processed_images})
        _, spatial_probs = sess.run([spatial_batch, spatial_probs])
        _, temporal_probs = sess.run([temporal_batch, temporal_probs])

    print spatial_probs
    print spatial_probs.shape, np.amax(spatial_probs)
    print temporal_probs.shape, np.amax(temporal_probs)


    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    # print_tensors_in_checkpoint_file(file_name='checkpoints/spatial_vgg16.ckpt', tensor_name=None, all_tensors=False)
    # print_tensors_in_checkpoint_file(file_name='checkpoints/temporal_vgg16.ckpt', tensor_name=None, all_tensors=False)