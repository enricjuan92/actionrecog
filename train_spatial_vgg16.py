import os, sys, time, glob, math, cv2
import urllib2 as urllib

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from video_processing import io
from datasets import imagenet, ucf101_utils
from models.vgg import vgg
from preprocessing import vgg_preprocessing
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

slim = tf.contrib.slim

# PATHs definitions
VIDEOS_PATH = 'resources/'
SPATIAL_CKPT = 'checkpoints/spatial_vgg16.ckpt'

with tf.Graph().as_default():

    log_dir = 'checkpoints_2/'  # Where checkpoints are stored

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)


    images, labels = ucf101_utils.load_train_dataset(100)
    # images = images[2000:2150]
    # labels = labels[2000:2150]
    print('Training set')
    print('\tImage tensor: ', images.shape)
    print('\tLabels tensor: ', labels.shape)

    # images = tf.cast(images, dtype=tf.float32)
    # labels = tf.cast(labels, dtype=tf.int32)

    # Create the model, fuse the default arg scope to configure the batch norm parameters
    with slim.arg_scope(vgg.vgg_arg_scope()):
        predictions, _ = vgg.vgg_16(images, num_classes=101, scope='spatial_vgg16')


    print('Logits tensor: ', predictions.shape)

    print('Specify the loss function:')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=predictions)

    total_loss = tf.losses.get_total_loss()
    print('Total loss: ', total_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    print('Computing gradient ...')
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # create_train_op ensures that each time we ask for the loss, the update_ops
    # are run and the gradients being computed are applied too
    print('Creating train operation ...')
    train_op = slim.learning.create_train_op(total_loss, optimizer)#, variables_to_train='spatial_vgg16/fc8/weights')

    # Get list of variables to restore
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['spatial_vgg16/fc8/weights',
                                                                                  'spatial_vgg16/fc8/biases'])
    # Add ops to restore all the variables.
    init_assign_op =  tf.contrib.framework.assign_from_checkpoint_fn(model_path=SPATIAL_CKPT,
                                                                     var_list=variables_to_restore,
                                                                     ignore_missing_vars=True)

    print('Restore variables from checkpoint.')

    # print_tensors_in_checkpoint_file(file_name='checkpoints/spatial_vgg16.ckpt', tensor_name=None, all_tensors=False)

    # with tf.Session() as sess:
    #     init_assign_op(sess)
    #
    #     with tf.variable_scope('spatial_vgg16/vgg_16/fc8'):
    #         print(tf.get_variable('weights'))

    print('Actually runs training:')
    init = tf.initialize_all_variables()

    final_loss = slim.learning.train(train_op,
                                     logdir=log_dir,
                                     init_fn=init,
                                     number_of_steps=1000,
                                     save_summaries_secs=300,
                                     save_interval_secs=600)

    print('Finished training. Last batch loss %f' % final_loss)

    # with tf.Session() as sess:
    #     init_assign_op(sess)
    # # Create an initial assignment function.
    # def InitAssignFn(sess):
    #     sess.run(init_assign_op, init_feed_dict)

    # Creates a variable to hold the global_step.
    # global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
    # tf.add_to_collection('global_step', global_step_tensor)
    # Creates a session.
    # sess = tf.Session()
    # Initializes the variable.
    # sess.run(global_step_tensor.initialized_value())
    # print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))