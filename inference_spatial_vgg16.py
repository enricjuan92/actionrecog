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

def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

with tf.Graph().as_default():

    # SET PATHS
    work_dir = '../work/ucf101_jpegs_256/jpegs_256/'

    # test_split_path = 'datasets/valid_datasets/ucf101_valid_split1.txt'
    test_split_path = 'datasets/valid_datasets/ucf101_testlist02.txt'

    # checkpoint_path = 'checkpoints/finetune_spatial_vgg16_11_06.ckpt'
    # checkpoint_path = 'checkpoints/finetune_spatial_vgg16_split1.ckpt'
    checkpoint_path = 'checkpoints/finetune_spatial_trainlist01.ckpt'
    # checkpoint_path = 'checkpoints/spatial_vgg16_1.ckpt'
    filewriter_path = 'tensorboard_spatial/'


    model_scope = 'spatial_vgg16'

    if not tf.gfile.Exists(filewriter_path):
        tf.gfile.MakeDirs(filewriter_path)

    # SET UP CONFIGURATION VARIABLES
    num_samples_per_clip = 25
    batch_size = 200
    batch_size = 50

    test_dataset_num_clips = 3750
    test_dataset_num_clips = 2000
    test_dataset_clips_per_split = 1 # test_dataset [50*20*10, 224, 224, 3]
    test_dataset_offset = 0

    display_step = 1

    # PLACEHOLDERS
    ph_images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
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

        pickle_predictions = np.zeros((test_dataset_num_clips, 101))
        pickle_labels = np.zeros((test_dataset_num_clips, 101))

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

            test_dataset, test_labels = ucf101_utils.load_validation_spatial_dataset(batch_size=test_dataset_clips_per_split,
                                                                                     offset=test_dataset_offset,
                                                                                     split_dir=test_split_path,
                                                                                     work_dir=work_dir,
                                                                                     num_samples=num_samples_per_clip)

            print('Validation subset #%d' % dataset_step)
            print('->Image subset: ', test_dataset.shape)
            print('->Labels subset: ', test_labels.shape)

            test_batches_per_epoch = np.floor(test_dataset.shape[0] / batch_size).astype(np.int16)

            itest_acc = 0.
            itest_count = 0
            spatial_prediction = np.zeros((250, 101))

            for step in range(test_batches_per_epoch):

                offset = (step * batch_size) % (test_labels.shape[0] - batch_size)
                start = offset
                end = (offset + batch_size)

                batch_data = test_dataset[start:end, :, :, :]
                batch_labels = test_labels[start:end, :]

                feed_dict = {ph_images: batch_data, ph_labels: batch_labels}

                b_start = (step * batch_size) % (200)
                b_end = b_start + batch_size

                spatial_prediction[b_start:b_end, :] = sess.run(scores, feed_dict=feed_dict)
                spatial_labels = batch_labels[0, :]

                acc = sess.run(accuracy, feed_dict=feed_dict)

                print('Calculating accuracy mean ... Step %d' % step)

                itest_acc += acc
                itest_count += 1

                if ((step + 1) % 5 == 0):
                    avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=0)
                    avg_spatial_pred = softmax(avg_spatial_pred_fc8)

                    pickle_predictions[dataset_step, :] = avg_spatial_pred
                    pickle_labels[dataset_step, :] = spatial_labels

                    avg_spatial_pred = np.expand_dims(avg_spatial_pred, axis=0)
                    spatial_labels = np.expand_dims(spatial_labels, axis=0)

                    print(avg_spatial_pred.shape)

                    video_acc = np_accuracy(avg_spatial_pred, spatial_labels)
                    print('NP temporal accuracy: ', video_acc)

            itest_acc /= itest_count
            print("Validation Mean Accuracy from subset #%d = %.1f%%" % ((dataset_step + 1), (itest_acc * 100)))

            test_acc += video_acc
            test_count += 1

        dict = {
            'dataset': pickle_predictions,
            'labels': pickle_labels
        }

        ucf101_utils.save_pickle('spatial_predictions.pickle', dict)

        test_acc /= test_count
        print('Number of batch accuracies: %d (test_count %d)' % ((test_batches_per_epoch * dataset_splits), test_count))
        print("Validation Mean Accuracy = %.1f%%" % (test_acc))