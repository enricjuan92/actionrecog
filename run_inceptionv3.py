import os
import sys
import urllib2 as urllib


import tensorflow as tf
# import numpy as np
from datasets import imagenet
from models.inception import inception_v3
from preprocessing import inception_preprocessing

slim = tf.contrib.slim

image_size = inception_v3.inception_v3.default_image_size

with tf.Graph().as_default():
    #url = sys.argv[1]

    url = 'http://cv-tricks.com/wp-content/uploads/2017/03/pexels-photo-361951.jpeg'
    image_string = urllib.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    checkpoints_dir = 'checkpoints'

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
        slim.get_model_variables('InceptionV3'))

    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

    names = imagenet.create_readable_names_for_imagenet_labels()
    result_text = ''

    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (100 * probabilities[index], names[index]))

    result_text += str(names[sorted_inds[0]]) + '=>' + str(
        "{0:.2f}".format(100 * probabilities[sorted_inds[0]])) + '%\n'

