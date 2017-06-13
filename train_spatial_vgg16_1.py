import math

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')

from datasets.UCF101 import ucf101_utils
from models.vgg import vgg
from datetime import datetime

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops_impl

slim = tf.contrib.slim

def np_accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Graph().as_default():

    # SET PATHS
    work_dir = '../work/ucf101_jpegs_256/jpegs_256/'

    train_split_path = 'datasets/train_datasets/ucf101_train_split1.txt'
    valid_split_path = 'datasets/valid_datasets/ucf101_valid_split1.txt'

    # checkpoint_path = 'checkpoints/spatial_vgg16.ckpt'
    checkpoint_path = 'checkpoints/finetune_spatial_vgg16_split1.ckpt'
    save_checkpoint_path = 'checkpoints/finetune_spatial_vgg16_split1.ckpt'

    filewriter_path = 'tensorboard_spatial/'

    if not tf.gfile.Exists(filewriter_path):
        tf.gfile.MakeDirs(filewriter_path)

    # SET UP CONFIGURATION VARIABLES
    train_layers = ['fc8']
    model_scope = 'spatial_vgg16'

    display_step = 1

    num_samples_per_clip = 1
    batch_size = 200
    num_epochs = 35

    dropout_ratio = 0.8
    keep_prob = 1 - dropout_ratio
    starter_learning_rate = 0.001
    decay_steps = 4000

    train_dataset_num_clips = 4000 # real = 200k frames -> 4k clips * 5 (samples per clip) * 10 (data aug.)
    train_dataset_batch_size = 4000 # train_dataset.shape [4000, 224, 224, 3]

    valid_dataset_num_clips = 440 # real = 44k -> 440 frames * 20 (#samples) * 10 (data aug.)
    valid_dataset_batch_size = 10 # valid_dataset.shape [2.000, 224, 224, 20]

    global_step = tf.Variable(0, trainable=False)

    # PLACEHOLDERS
    ph_dataset = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name='ph_dataset')
    ph_labels = tf.placeholder(tf.int32, [batch_size, 101], name='ph_labels')

    learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=0.96,
                                               staircase=True)

    # Create the model
    with slim.arg_scope(vgg.vgg_arg_scope()):
        # TRAINING
        scores, _ = vgg.vgg_16(inputs=ph_dataset,
                               num_classes=101,
                               dropout_keep_prob=keep_prob,
                               is_training=True,
                               scope=model_scope)
        probabilities = tf.nn.softmax(logits=scores)

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers]
    print(var_list)
    # Op for calculating the loss
    with tf.name_scope('cross_ent'):

        print('Specify the loss function:')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ph_labels, logits=scores))
        print(scores.shape, loss.shape)
        # loss = tf.losses.softmax_cross_entropy(onehot_labels=ph_labels, logits=scores)
        total_loss = tf.losses.get_total_loss()

    # Add the loss to summary
    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    # Train op: specify the optimization scheme
    with tf.name_scope('train'):

        print('Computing gradient ...')
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(loss=loss,
                                      var_list=var_list,
                                      global_step=global_step)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Get list of variables to restore
    # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=train_layers)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=[])

    # Add ops to restore all the variables.
    init_assign_op =  tf.contrib.framework.assign_from_checkpoint_fn(model_path=checkpoint_path,
                                                                     var_list=variables_to_restore,
                                                                     ignore_missing_vars=True)
    print('Restore variables from checkpoint.')

    # init_var_op = tf.variables_initializer(var_list=var_list)
    # Evaluation op: Accuracy of the model
    correct_pred = tf.equal(tf.argmax(probabilities, 1), tf.argmax(ph_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Initialize all variables
        # tf.global_variables_initializer().run()

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        init_assign_op(sess)
        print('Model initialized.')
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

        for epoch in range(num_epochs):

            dataset_batchs = np.floor(train_dataset_num_clips / train_dataset_batch_size).astype(np.int16)
            print('Number of dataset batches: %d' % dataset_batchs)

            for dataset_step in range(dataset_batchs):

                # TRAINING
                train_dataset_offset = (dataset_step * train_dataset_batch_size) % \
                                       (train_dataset_num_clips - train_dataset_batch_size)

                print('-> train_dataset from %d to %d' % \
                      (train_dataset_offset, train_dataset_offset + train_dataset_batch_size))

                # Array allocation
                train_dataset, train_labels = ucf101_utils.load_train_spatial_dataset(batch_size=train_dataset_batch_size,
                                                                                      offset=train_dataset_offset,
                                                                                      split_dir=train_split_path,
                                                                                      work_dir=work_dir,
                                                                                      num_samples=num_samples_per_clip)

                # train_dataset [4000, 224, 224, 3]
                train_dataset, train_labels = ucf101_utils.randomize(train_dataset, train_labels)

                train_batches_per_epoch = np.floor(train_dataset.shape[0] / batch_size).astype(np.int16)
                print('Train batches per epoch: %d' % train_batches_per_epoch)

                print('Training set')
                print('Image tensor: ', train_dataset.shape)
                print('Labels tensor: ', train_labels.shape)

                print("{} Start Training".format(datetime.now()))

                for step in range(train_batches_per_epoch):

                    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                    start = offset
                    end = (start + batch_size)

                    # Generate a minibatch.
                    batch_data = train_dataset[start:end, :, :, :]
                    batch_labels = train_labels[start:end, :]

                    feed_dict = {ph_dataset: batch_data, ph_labels: batch_labels}

                    lr, _, l, predictions = sess.run([learning_rate, train_op, loss, scores], feed_dict=feed_dict)

                    if ((step + 1) % display_step == 0 and step != 0):

                        s = sess.run(merged_summary, feed_dict=feed_dict)
                        writer.add_summary(s, epoch * train_batches_per_epoch + (step + 1))

                    if ((step + 1) % 10 == 0):

                        print('Learning rate: %.12f' % lr)
                        print('Epoch: %d. Step: %d From: %d To: %d' % (epoch, (step + 1), start, end))
                        print("Minibatch loss at step %d: %f" % ((step + 1), l))
                        print("Minibatch accuracy: %.1f%%" %  np_accuracy(predictions, batch_labels))

                # if dataset_step == (dataset_batchs - 1):
                #
                #     # VALIDATION
                #     print("{} Start validation".format(datetime.now()))
                #     test_acc = 0.
                #     test_count = 0
                #
                #     # valid_dataset_offset = (dataset_step * valid_dataset_batch_size) % \
                #     #                        (valid_dataset_num_clips - valid_dataset_batch_size)
                #     valid_dataset_offset = 0
                #
                #     print('valid_dataset from %d to %d' % \
                #           (valid_dataset_offset, valid_dataset_offset + valid_dataset_batch_size))
                #
                #     valid_dataset, valid_labels = ucf101_utils.load_spatial_dataset(batch_size=valid_dataset_batch_size,
                #                                                                     offset=valid_dataset_offset,
                #                                                                     split_dir=valid_split_path,
                #                                                                     work_dir=work_dir,
                #                                                                     num_samples=num_samples_per_clip)
                #     print('Validation set')
                #     print('Image tensor: ', valid_dataset.shape)
                #     print('Labels tensor: ', valid_labels.shape)
                #
                #     valid_batches_per_epoch = np.floor(valid_dataset.shape[0] / batch_size).astype(np.int16)
                #     print('Valid batches per epoch: %d' % valid_batches_per_epoch)
                #
                #     for step in range(valid_batches_per_epoch):
                #
                #         offset = (step * batch_size) % (valid_labels.shape[0] - batch_size)
                #         start = offset
                #         end = (offset + batch_size)
                #
                #         batch_data = valid_dataset[start:end, :, :, :]
                #         batch_labels = valid_labels[start:end, :]
                #
                #         feed_dict = {ph_dataset: batch_data, ph_labels: batch_labels}
                #         acc = sess.run(accuracy, feed_dict=feed_dict)
                #
                #         test_acc += acc
                #         test_count += 1
                #
                #     test_acc /= test_count
                #     print("Mean Validation Accuracy = %.1f%%" % (test_acc * 100))

                save_path = saver.save(sess, save_checkpoint_path)
                print("{} Model checkpoint saved at {}".format(datetime.now(), save_checkpoint_path))

    print('Finished training. Last batch loss %f' % l)
