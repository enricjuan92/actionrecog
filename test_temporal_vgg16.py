########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from labels import imagenet_classes

slim = tf.contrib.slim

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.parameters = []
        # imgs = tf.expand_dims(imgs, 0)
        # imgs = tf.image.rgb_to_grayscale(imgs)
        # imgs = tf.concat([imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs,
        #                   imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs, imgs], 3)
        # self.imgs = tf.cast(imgs, tf.float32)

        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        # self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs#-mean

        # conv1_1
        with tf.name_scope('temporal_vgg16/conv1/conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 20, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('temporal_vgg16/conv1/conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('temporal_vgg16/conv2/conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('temporal_vgg16/conv2/conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('temporal_vgg16/conv3/conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('temporal_vgg16/conv3/conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('temporal_vgg16/conv3/conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('temporal_vgg16/conv4/conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('temporal_vgg16/conv4/conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('temporal_vgg16/conv4/conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('temporal_vgg16/conv5/conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('temporal_vgg16/conv5/conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('temporal_vgg16/conv5/conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):

        # fc1
        shape = int(np.prod(self.pool5.get_shape()[1:]))

        with tf.name_scope('temporal_vgg16/fc6') as scope:
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1), name='old_weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')

        pool5_flat = tf.reshape(self.pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        self.fc1 = tf.nn.relu(fc1l)
        self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('temporal_vgg16/fc7') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='old_weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')

        fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        self.fc2 = tf.nn.relu(fc2l)
        self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('temporal_vgg16/fc8') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 101], dtype=tf.float32, stddev=1e-1), name='old_weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[101], dtype=tf.float32), trainable=True, name='biases')

        self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
        self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):

        weights = np.load(weight_file)
        items = weights[()].items()

        for i, j in enumerate(sorted(items)):
            print i, j[0], j[1].keys()[0], j[1][('weights')].shape

            i *= 2
            print i, i+1
            print self.parameters[i]
            print self.parameters[i+1]

            sess.run(self.parameters[i].assign(j[1]['weights']))
            sess.run(self.parameters[i+1].assign(j[1]['biases']))

            if i == 26:
                print self.parameters[i]
                self.parameters[i] = tf.Variable(tf.reshape(self.parameters[i], [7, 7, 512, 4096]), name='temporal_vgg16/fc6/weights')

                fc6w = np.reshape(j[1]['weights'], (7, 7, 512, 4096))
                print fc6w.shape

                sess.run(self.parameters[i].assign(fc6w))
                print self.parameters[i]

            if i == 28:
                print self.parameters[i]
                self.parameters[i] = tf.Variable(tf.reshape(self.parameters[i], [1, 1, 4096, 4096]), name='temporal_vgg16/fc7/weights')

                fc7w = np.reshape(j[1]['weights'], (1, 1, 4096, 4096))
                print fc7w.shape

                sess.run(self.parameters[i].assign(fc7w))
                print self.parameters[i]

            if i == 30:
                print self.parameters[i]
                self.parameters[i] = tf.Variable(tf.reshape(self.parameters[i], [1, 1, 4096, 101]), name='temporal_vgg16/fc8/weights')

                fc8w = np.reshape(j[1]['weights'], (1, 1, 4096, 101))
                print fc8w.shape

                sess.run(self.parameters[i].assign(fc8w))
                print self.parameters[i]


        saver = tf.train.Saver()
        saver.save(sess, 'checkpoints/temporal_vgg16.ckpt')
        #
        # for i in range(len(items)):
        #     pass
            # print self.parameters[i]
            # sess.run(self.parameters[i].assign(weights[k]))

        # weights = np.load('vgg16_weights.npz')
        # keys = sorted(weights.keys())
        # for i, k in enumerate(keys):
        #     print i, k, np.shape(weights[k])
        #     sess.run(self.parameters[i].assign(weights[k]))

if __name__ == '__main__':

    # sess = tf.Session()
    # imgs = tf.placeholder(tf.float32, [None, 224, 224, 20])
    # vgg = vgg16(imgs, 'temporal_vgg16.npy', sess)

    img1 = imread('resources/frame000044.jpg', mode='RGB')
    img1 = imresize(img1, (224, 224))

    # Add ops to restore all the variables.
    # weights1_1 = tf.Variable(tf.random_normal(shape=[3,3,20,64]),dtype=tf.float32_ref, name='')
    # Add ops to save and restore all the variables.

    img1 = tf.expand_dims(img1, 0)
    # print img1.shape
    img1 = tf.concat([img1, img1], 0)
    img1 = tf.image.rgb_to_grayscale(img1)

    img1 = tf.concat([img1, img1, img1, img1, img1, img1, img1, img1, img1, img1,
                      img1, img1, img1, img1, img1, img1, img1, img1, img1, img1], 3)
    img1 = tf.cast(img1, tf.float32)
    # img = np.copy(img1)
    # print img.shape
    with tf.Session() as sess:
        vgg = vgg16(img1, 'checkpoints/temporal_vgg16.npy', sess)
    #_, prob = sess.run(vgg.probs, feed_dict={vgg.imgs: img1})#[0]
        _, prob = sess.run([img1, vgg.probs])  # [0]

    print prob.shape
    # prob = np.squeeze(prob)
    # print prob.shape
    #
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print class_names[p], prob[p]

    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(file_name='checkpoints/temporal_vgg16.ckpt', tensor_name=None, all_tensors=False)