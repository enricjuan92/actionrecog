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

slim = tf.contrib.slim

# PATHs definitions
VIDEOS_PATH = 'resources/'
SPATIAL_CKPT = 'checkpoints/spatial_vgg16.ckpt'
TEMPORAL_CKPT = 'checkpoints/temporal_vgg16.ckpt'


with tf.Graph().as_default():

    spatial_ph = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    temporal_ph = tf.placeholder(shape=[None, 224, 224, 20], dtype=tf.float32)

    image = imread('resources/rgb_frames/frame000010.jpg', mode='RGB')
    rs_image = imresize(image, (224, 224))
    mean_rgb_image = np.mean(rs_image)
    rgb_image = tf.expand_dims(rs_image, 0)

    ####### FLOW PROCESSING #######

    num_frames = 0
    num_samples = 1
    optical_flow_frames = 10
    start_frame = 4
    vid_name = 'resources/flow_frames/'

    if num_frames == 0:
        imglist = glob.glob(os.path.join(vid_name, 'flow_u*.jpg'))
        duration = len(imglist)
        # print 'Image List: ', imglist, 'List length: ', duration
    else:
        duration = num_frames

        # selection
    step = int(math.floor((duration - optical_flow_frames + 1) / num_samples))

    dims = (224, 224, optical_flow_frames * 2, num_samples)
    flow = np.zeros(shape=dims, dtype=np.float64)
    flow_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        for j in range(optical_flow_frames):

            flow_x_file = os.path.join(vid_name, 'flow_u_frame{0:06d}.jpg'.format(i * step + j + 1 + start_frame))
            flow_y_file = os.path.join(vid_name, 'flow_v_frame{0:06d}.jpg'.format(i * step + j + 1 + start_frame))

            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)

            img_x = cv2.resize(img_x, dims[1::-1])
            img_y = cv2.resize(img_y, dims[1::-1])

            flow[:, :, j * 2, i] = img_x
            flow[:, :, j * 2 + 1, i] = img_y

            flow_flip[:, :, j * 2, i] = 255 - img_x[:, ::-1]
            flow_flip[:, :, j * 2 + 1, i] = img_y[:, ::-1]

    # crop
    print flow.shape

    flow_1 = flow[:224, :224, :, :]
    flow_2 = flow[:224, -224:, :, :]
    flow_3 = flow[:224, -224:, :, :]
    flow_4 = flow[-224:, :224, :, :]
    flow_5 = flow[-224:, -224:, :, :]
    flow_f_1 = flow_flip[:224, :224, :, :]
    flow_f_2 = flow_flip[:224, -224:, :, :]
    flow_f_3 = flow_flip[:224, -224:, :, :]
    flow_f_4 = flow_flip[-224:, :224, :, :]
    flow_f_5 = flow_flip[-224:, -224:, :, :]

    # print flow_1.shape, flow_2.shape, flow_3.shape, flow_4.shape, flow_5.shape

    flow = np.concatenate((flow_1, flow_2, flow_3, flow_4, flow_5, flow_f_1, flow_f_2, flow_f_3, flow_f_4, flow_f_5),
                          axis=3)

    # substract mean
    flow_mean = mean_rgb_image

    flow = flow - np.tile(flow_mean[..., np.newaxis], (1, 1, 1, flow.shape[3]))

    print flow.shape
    flow = np.transpose(flow, (3, 0, 1, 2))
    print flow.shape
    # gray_image = tf.image.rgb_to_grayscale(rgb_image)
    # stack_image = tf.concat([gray_image, gray_image, gray_image, gray_image, gray_image,
    #                         gray_image, gray_image, gray_image, gray_image, gray_image,
    #                         gray_image, gray_image, gray_image, gray_image, gray_image,
    #                         gray_image, gray_image, gray_image, gray_image, gray_image], 3)

    stack_image = flow

    rgb_image = np.random.randint(low=0, high=255, size=[1,224,224,3])
    stack_image = np.random.randint(low=0, high=255, size=[1, 224, 224, 20])

    spatial_batch = tf.cast(rgb_image, tf.float32)

    spatial_batch = tf.image.resize_images(images=image, size=[224, 224])
    spatial_batch = tf.expand_dims(spatial_batch, 0)
    print spatial_batch.shape
    temporal_batch = tf.cast(stack_image, tf.float32)

    # Create the model, fuse the default arg scope to configure the batch norm parameters
    with slim.arg_scope(vgg.vgg_arg_scope()):
        spatial_logits, _ = vgg.vgg_16(spatial_batch, num_classes=1000, is_training=False, scope='vgg_16')
        # temporal_logits, _ = vgg.vgg_16(temporal_batch, num_classes=101, is_training=False, scope='temporal_vgg16')

    spatial_probs = tf.nn.softmax(spatial_logits)
    # temporal_probs = tf.nn.softmax(temporal_logits)

    # Define the loss functions and get the total loss.
    # loss = slim.losses.softmax_cross_entropy(logits, labels)
    # total_loss = slim.losses.get_total_loss()
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # create_train_op ensures that each time we ask for the loss, the update_ops
    # are run and the gradients being computed are applied too
    # train_op = slim.learning.create_train_op(total_loss, optimizer)
    # logdir = ...  # Where checkpoints are stored


    # Add ops to restore all the variables.
    spatial_init = tf.contrib.framework.assign_from_checkpoint_fn(model_path='checkpoints/imagenet_vgg16.ckpt',
                                                                  var_list=slim.get_model_variables(),
                                                                  ignore_missing_vars=True)
    # temporal_init = tf.contrib.framework.assign_from_checkpoint_fn(model_path=TEMPORAL_CKPT,
    #                                                               var_list=slim.get_model_variables(),
    #                                                               ignore_missing_vars=True)

    # slim.learning.train(train_op, log_dir, init_fn=init_fn)

    with tf.Session() as sess:  # set up the session

        spatial_init(sess)
        # temporal_init(sess)

        # np_images, probabilities = sess.run(probabilities, feed_dict={x: processed_images})
        _, spatial_probs = sess.run([spatial_batch, spatial_probs])
        # _, temporal_probs = sess.run([temporal_batch, temporal_probs])

    print spatial_probs.shape, np.amax(spatial_probs)*100
    # print temporal_probs.shape, np.amax(temporal_probs)


    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    # print_tensors_in_checkpoint_file(file_name='checkpoints/spatial_vgg16.ckpt', tensor_name=None, all_tensors=False)
    # print_tensors_in_checkpoint_file(file_name='checkpoints/temporal_vgg16.ckpt', tensor_name=None, all_tensors=False)