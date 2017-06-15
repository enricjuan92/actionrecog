import math
import matplotlib
import numpy as np
import os
import tensorflow as tf

matplotlib.use('Agg')

from datasets.UCF101 import ucf101_utils
from models.vgg import vgg
from datetime import datetime

slim = tf.contrib.slim

def np_accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Graph().as_default():

    # SET PATHS
    work_dir = '../work/ucf101_tvl1_flow/tvl1_flow/'

    # test_split_path = 'datasets/valid_datasets/ucf101_valid_split1.txt'
    test_split_path = 'datasets/valid_datasets/ucf101_testlist01.txt'

    # checkpoint_path = 'checkpoints/finetune_spatial_vgg16_11_06.ckpt'
    # checkpoint_path = 'checkpoints/finetune_spatial_vgg16_split1.ckpt'
    checkpoint_path = 'checkpoints/finetune_temporal_trainlist01_1.ckpt'
    filewriter_path = 'tensorboard_temporal/'


    model_scope = 'vgg_16'

    if not tf.gfile.Exists(filewriter_path):
        tf.gfile.MakeDirs(filewriter_path)

    # SET UP CONFIGURATION VARIABLES
    num_samples_per_clip = 18
    batch_size = 180

    test_dataset_num_clips = 100
    test_dataset_clips_per_split = 10 # test_dataset [10*20*10, 224, 224, 3]
    test_dataset_offset = 0

    display_step = 1

    # PLACEHOLDERS
    ph_images = tf.placeholder(tf.float32, [batch_size, 224, 224, 20])
    ph_labels = tf.placeholder(tf.float32, [batch_size, 101])

    # Create the model
    with slim.arg_scope(vgg.vgg_arg_scope()):
        scores, _ = vgg.vgg_16(ph_images, num_classes=101, is_training=False, scope=model_scope)
        # scores.shape [10, 101] -> avg_scores.shape [1, 101]
        avg_scores = tf.reduce_mean(input_tensor=scores, axis=0)
        # labels [10, 101] -> avg_labels [1, 101]
        avg_labels = tf.reduce_mean(input_tensor=ph_labels, axis=0)
        # probabilities.shape [1, 101]
        probabilities = tf.nn.softmax(logits=avg_scores)

    # Get list of variables to restore
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=[])

    # Add ops to restore all the variables.
    init_assign_op =  tf.contrib.framework.assign_from_checkpoint_fn(model_path=checkpoint_path,
                                                                     var_list=variables_to_restore,
                                                                     ignore_missing_vars=True)
    print('Restore variables from checkpoint.')

    # Evaluation op: Accuracy of the model

    correct_pred = tf.equal(tf.argmax(tf.expand_dims(probabilities, axis=0), 1),
                            tf.argmax(tf.expand_dims(avg_labels, axis=0), 1))

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

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        init_assign_op(sess)

        print('Model initialized.')
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))
        print("{} Start validation".format(datetime.now()))

        test_acc = 0.
        test_count = 0

        dataset_splits = np.floor(test_dataset_num_clips / test_dataset_clips_per_split).astype(np.int16)
        print('Number of dataset batches: %d' % dataset_splits)

        for dataset_step in range(dataset_splits):

            test_dataset_offset = (dataset_step * test_dataset_clips_per_split) % \
                                   (test_dataset_num_clips - test_dataset_clips_per_split)

            test_dataset, test_labels = ucf101_utils.load_validation_flow_dataset(batch_size=test_dataset_clips_per_split,
                                                                                  offset=test_dataset_offset,
                                                                                  split_dir=test_split_path,
                                                                                  work_dir=work_dir,
                                                                                  num_samples=num_samples_per_clip)

            print('Validation subset #%d' % (dataset_step + 1))
            print('->Image subset: ', test_dataset.shape)
            print('->Labels subset: ', test_labels.shape)

            test_batches_per_epoch = np.floor(test_dataset.shape[0] / batch_size).astype(np.int16)

            itest_acc = 0.
            itest_count = 0

            for step in range(test_batches_per_epoch):

                offset = (step * batch_size) % (test_labels.shape[0] - batch_size)
                start = offset
                end = (offset + batch_size)

                batch_data = test_dataset[start:end, :, :, :]
                batch_labels = test_labels[start:end, :]

                feed_dict = {ph_images: batch_data, ph_labels: batch_labels}

                acc = sess.run(accuracy, feed_dict=feed_dict)

                print('Calculating accuracy mean ... Step %d' % step)

                itest_acc += acc
                itest_count += 1

                test_acc += acc
                test_count += 1

            itest_acc /= itest_count
            print("Validation Mean Accuracy from subset #%d = %.1f%%" % ((dataset_step + 1), (itest_acc * 100)))

        test_acc /= test_count
        print('Number of batch accuracies: %d (test_count %d)' % ((test_batches_per_epoch*dataset_splits), test_count))
        print("Validation Mean Accuracy = %.1f%%" % (test_acc * 100))