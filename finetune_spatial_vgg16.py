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
from datetime import datetime
from preprocessing import vgg_preprocessing
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

slim = tf.contrib.slim

# PATHs definitions
VIDEOS_PATH = 'resources/'
SPATIAL_CKPT = 'checkpoints/spatial_vgg16.ckpt'

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Graph().as_default():

    batch_size = 50

    filewriter_path = 'tensorboard/'
    if not tf.gfile.Exists(filewriter_path):
        tf.gfile.MakeDirs(filewriter_path)

    images, labels = ucf101_utils.load_train_dataset(5000)
    # images = images[2000:2150]
    # labels = labels[2000:2150]
    print('Training set')
    print('\tImage tensor: ', images.shape)
    print('\tLabels tensor: ', labels.shape)

    ph_images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    ph_labels = tf.placeholder(tf.float32, [batch_size, 101])

    fc8w = tf.Variable(tf.random_normal(shape=[1, 1, 4096, 101], stddev=0.01), name='spatial_vgg16/fc8/weights')
    fc8b = tf.Variable(tf.random_normal(shape=[101], stddev=0.01), name='spatial_vgg16/fc8/biases')


    # Create the model, fuse the default arg scope to configure the batch norm parameters
    with slim.arg_scope(vgg.vgg_arg_scope()):
        predictions, _ = vgg.vgg_16(ph_images, num_classes=101, scope='spatial_vgg16')
        train_prediction = tf.nn.softmax(predictions)

    print('Specify the loss function:')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=ph_labels, logits=predictions)
    total_loss = tf.losses.get_total_loss()

    # Add the loss to summary
    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    print('Computing gradient ...')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(total_loss)

    # Get list of variables to restore
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['spatial_vgg16/fc8/weights',
                                                                                  'spatial_vgg16/fc8/biases'])

    # Add ops to restore all the variables.
    init_assign_op =  tf.contrib.framework.assign_from_checkpoint_fn(model_path=SPATIAL_CKPT,
                                                                     var_list=variables_to_restore,
                                                                     ignore_missing_vars=True)
    print('Restore variables from checkpoint.')
    init_var_op = tf.variables_initializer(var_list=[fc8w, fc8b])

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    num_steps = 9500

    train_batches_per_epoch = np.floor(len(images) / batch_size).astype(np.int16)
    # num_steps = train_batches_per_epoch

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        print('Fc8 layer initialized.')

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        init_assign_op(sess)
        print('Model initialized.')

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

        for step in range(num_steps):

            offset = (step * batch_size) % (labels.shape[0] - batch_size)
            start = offset
            end = (offset + batch_size)

            # Generate a minibatch.
            batch_data = images[start:end, :]
            batch_labels = labels[start:end, :]

            feed_dict = {ph_images: batch_data, ph_labels: batch_labels}

            _, l, predictions = sess.run([optimizer, total_loss, train_prediction], feed_dict=feed_dict)

            # Generate summary with the current batch of data and write to file
            if step % 1 == 0:
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, train_batches_per_epoch + step)

            if (step % 25 == 0):
                print('Step: %d From: %d To: %d' % (step, start, end))
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

            if (step % 500 == 0):
                checkpoint_name = 'checkpoints/finetuning_spatial_vgg16_2.ckpt'
                save_path = saver.save(sess, checkpoint_name)
                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


    print('Finished training. Last batch loss %f' % l)
