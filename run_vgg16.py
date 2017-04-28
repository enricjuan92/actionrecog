import os, sys, time
import urllib2 as urllib

import tensorflow as tf
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from video_processing import io

from datasets import imagenet
from models.vgg import vgg
from preprocessing import vgg_preprocessing
# from preprocessing import inception_preprocessing

slim = tf.contrib.slim

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# PATHs definitions
VIDEOS_PATH = 'resources/'
CKPT_PATH = 'checkpoints/vgg_16.ckpt'

videos = os.listdir(VIDEOS_PATH)
video_in = os.path.join(VIDEOS_PATH, videos[0])
print(video_in)


image_size = vgg.vgg_16.default_image_size
labels = imagenet.create_readable_names_for_imagenet_labels()
del labels[0]


with tf.Graph().as_default():
    # images = tf.placeholder(tf.int8, [32, 224, 224, 3], name="input_frames")  # Our input is 10x10

    # video [channels, frames, height, width]
    # tensor [frames, height, width, channels]
    # t_images = io.video_to_tensor(npvideo)

    # npframes = io.video_to_array(video_path=video_in,
    #                              resize=[image_size, image_size],
    #                              start_frame=200,
    #                              end_frame=201)
    # npframes = io.video_to_array(video_path=video_in,
    #                              start_frame=200,
    #                              end_frame=201)
    # print(npframes.shape)
    #
    # t_image = io.video_to_tensor(npframes)
    # print(t_image.shape)
    # frame = tf.squeeze(t_image)
    # print(frame.shape)

    x = tf.placeholder(shape=[2, 224, 224, 3], dtype=tf.float32)

    url1 = 'http://cv-tricks.com/wp-content/uploads/2017/03/pexels-photo-361951.jpeg'
    url2 = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'

    image1_string = urllib.urlopen(url1).read()
    image2_string = urllib.urlopen(url2).read()

    image1 = tf.image.decode_jpeg(image1_string, channels=3)
    image2 = tf.image.decode_jpeg(image2_string, channels=3)

    images = tf.concat([image1, image2], 0)

    processed_image1 = vgg_preprocessing.preprocess_image(image1, image_size, image_size, is_training=False)
    processed_image2 = vgg_preprocessing.preprocess_image(image2, image_size, image_size, is_training=False)

    processed_image1 = tf.expand_dims(processed_image1, 0)
    processed_image2 = tf.expand_dims(processed_image2, 0)

    processed_images = tf.concat([processed_image1, processed_image2], 0)

    print(processed_images.shape)


    # Create the model, fuse the default arg scope to configure the batch norm parameters
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)

    # Define the loss functions and get the total loss.
    # loss = slim.losses.softmax_cross_entropy(logits, labels)
    # total_loss = slim.losses.get_total_loss()
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # create_train_op ensures that each time we ask for the loss, the update_ops
    # are run and the gradients being computed are applied too
    # train_op = slim.learning.create_train_op(total_loss, optimizer)
    # logdir = ...  # Where checkpoints are stored

    print logits, probabilities

    # Add ops to restore all the variables.
    variables_to_restore = slim.get_variables_to_restore(exclude=['fc8'])
    # init_fn = tf.contrib.framework.assign_from_checkpoint_fn(CKPT_PATH, slim.get_model_variables())
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(CKPT_PATH, variables_to_restore)

    # slim.learning.train(train_op, log_dir, init_fn=init_fn)

    with tf.Session() as sess:  # set up the session
        # plt.figure()
        # plt.imshow(frame.eval())
        # plt.savefig('processed_image')
        init_fn(sess)
        feed_dict = [processed_image1, processed_image2]
        # np_images, probabilities = sess.run(probabilities, feed_dict={x: processed_images})
        np_images, probabilities = sess.run([processed_images, probabilities])

        # print('Test accuracy: %.1f%%' % accuracy(probabilities.eval(), labels))
    print np_images.shape, probabilities.shape

    probabilities_0 = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities_0), key=lambda x: x[1])]

    for i in range(5):
        index = sorted_inds[i]
        print('Probability_0 %0.2f%% => [%s]' % (100 * probabilities_0[index], labels[index]))

    probabilities_1 = probabilities[1, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities_1), key=lambda x: x[1])]

    for i in range(5):
        index = sorted_inds[i]
        print('Probability_1 %0.2f%% => [%s]' % (100 * probabilities_1[index], labels[index]))


    plt.figure()
    plt.imshow(np_images[0].astype(np.uint8))
    plt.savefig('results/result1')

    plt.figure()
    plt.imshow(np_images[1].astype(np.uint8))
    plt.savefig('results/result2')