from __future__ import print_function

import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from scipy.misc import imresize
from six.moves import cPickle as pickle
import vgg_preprocessing

np.random.seed(133)

import tensorflow as tf


def maybe_extract(filename, force=False):

    data_root = '.'
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz, .zip, .rar, ...

    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))

    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()

    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    # if len(data_folders) != num_classes:
    #     raise Exception(
    #         'Expected %d folders, one per class. Found %d instead.' % (
    #             num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


def load_class(folder, mode='RGB', width=224, height=224):
    """Load the data for a single letter label."""
    pixel_depth = 255.0  # Number of levels per pixel.

    if mode == 'RGB':
        channels = 3

    image_files = os.listdir(folder)

    if mode == 'RGB':
        dataset = np.ndarray(shape=(len(image_files), width, height, channels), dtype=np.float32)
    elif mode == 'GRAY':
        dataset = np.ndarray(shape=(len(image_files), width, height), dtype=np.float32)
    else:
        raise Exception('Input not recognized. MODE argument must be RGB or GRAY')

    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)

        try:
            image_data = (ndimage.imread(image_file).astype(float))# - pixel_depth / 2) / pixel_depth
            image_data = imresize(image_data, (width, height))

            # if image_data.shape != (image_size, image_size):
            #     raise Exception('Unexpected image shape: %s' % str(image_data.shape), ' ', image_file)

            if mode == 'RGB':
                dataset[num_images, :, :, :] = image_data
            else:
                dataset[num_images, :, :] = image_data

            num_images = num_images + 1

        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    if mode == 'RGB':
        dataset = dataset[0:num_images, :, :, :]
    else:
        dataset = dataset[0:num_images, :, :]

    print('Full dataset tensor:', dataset.shape)
    # print('Mean:', np.mean(dataset))
    # print('Standard deviation:', np.std(dataset))
    return dataset


def load_frame(file, mode='RGB', width=224, height=224):
    """Load the data for a single letter label."""
    pixel_depth = 255.0  # Number of levels per pixel.

    if mode == 'RGB':
        channels = 3

    if mode == 'RGB':
        dataset = np.ndarray(shape=(width, height, channels), dtype=np.float32)
    elif mode == 'GRAY':
        dataset = np.ndarray(shape=(width, height), dtype=np.float32)
    else:
        raise Exception('Input not recognized. MODE argument must be RGB or GRAY')

    try:
        image_data = (ndimage.imread(file).astype(float))# - pixel_depth / 2) / pixel_depth
        image_data = imresize(image_data, (width, height))

        # if image_data.shape != (image_size, image_size):
        #     raise Exception('Unexpected image shape: %s' % str(image_data.shape), ' ', image_file)

        if mode == 'RGB':
            dataset[:, :, :] = image_data
        else:
            dataset[:, :] = image_data

    except IOError as e:
        print('Could not read:', file, ':', e, '- it\'s ok, skipping.')

    if mode == 'RGB':
        dataset = dataset[:, :, :]
    else:
        dataset = dataset[:, :]

    print('Full dataset tensor:', dataset.shape)
    # print('Mean:', np.mean(dataset))
    # print('Standard deviation:', np.std(dataset))
    return dataset


def create_pickle(data_folders, mode='RGB', force=False):

    dataset_names = []

    for folder in data_folders:
        set_filename = os.path.splitext(folder)[0] + '.pickle'
        dataset_names.append(set_filename)

        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_frame(folder, mode)

            try:
                filename = set_filename.split('/')[-1]

                with open(os.path.join('../work/train/ucf101_split1/', filename), 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

def create_ucf101_labels():
    with open('datasets/ucf101_splits/classInd.txt') as file:
        lines = file.readlines()

    labels = {0: 'background'}

    for line in lines:
        label = line.split(' ')
        labels[int(label[0])] = label[1].split('\n')[0]

    del labels[0]
    return labels


def make_arrays(rows, image_width, image_height, channels, num_classes=101):

    if rows:
        dataset = np.ndarray([rows, image_width, image_height, channels], dtype=np.float32)
        labels = np.ndarray([rows, num_classes], dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def rewrite_split_file():
    with open('datasets/ucf101_splits/train_rgb_split1.txt') as file:
        lines = file.readlines()

    print('File_split read. Length: ', len(lines), ' lines')

    new_file = []
    for num, line in enumerate(lines):

        folder = (line.split(' ')[0]).split('/')[-1]
        frame = line.split(' ')[1]
        label = (line.split(' ')[2]).split('\n')[0]

        print(folder, frame, label)
        new_path = os.path.join('../work/ucf101_jpegs_256/jpegs_256/', folder)
        new_line = os.path.join(new_path, 'frame{0:06d}.jpg'.format(int(frame)))

        if not os.path.isfile(new_line.split('\n')[0]):
            new_line = os.path.join(new_path,'frame{0:06d}.jpg'.format(np.random.randint(1,10)))

            if not os.path.isfile(new_line):
                raise Exception('Folder not existing: %s' % folder)

        new_line = new_line + ' ' + label + '\n'
        new_file.append(new_line)
        print(new_line, os.path.isfile(new_line.split(' ')[0]), num)

    print('File split reformat. ' ,'Length: ', len(new_file))

    with open('datasets/ucf101_splits/train_rgb_split_2.txt', 'wr+') as file:
        file.writelines(new_file)

# rewrite_split_file()

def read_split(split_dir):

    dataset = []
    labels  = []

    with open(split_dir) as file:
        lines = file.readlines()

        for line in lines:
            frame = line.split(' ')[0]
            label = line.split(' ')[1].split('\n')[0]

            dataset.append(frame)
            labels.append(label)

    return dataset, labels

def read_train_dataset(split_dir, work_dir):

    train_frames = []
    train_labels = []

    with open(split_dir) as file:
        lines = file.readlines()

    print('Split_file length: ', len(lines))

    for line in lines:
        line = line.split('\n')[0]

        frame_file = line
        label_folder = line.split('/')[-2].split('_')[1]

        train_frames.append(frame_file)
        train_labels.append(label_folder)

    return train_frames, train_labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation,:]

    return shuffled_dataset, shuffled_labels

def save_pickle(pickle_file, save):
    # pickle_file = os.path.join('../work/', 'ucf101_rgb_train.pickle')

    try:
        f = open(pickle_file, 'wb')
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

def load_pickle(pickle_file):

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        dataset = save['dataset']
        labels = save['labels']
        del save  # hint to help gc free up memory

    print('Dataset dimensions: ', dataset.shape, labels.shape)

    return dataset, labels

def generate_pickle_dataset(batch_size, split_dir, pickle_dir, is_training=False):

    dataset, labels = read_split(split_dir)

    try:
        f_dataset = dataset[0:batch_size]
        f_labels  = labels[0:batch_size]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    np_dataset, np_labels = make_arrays(len(f_dataset),
                                        image_width=224,
                                        image_height=224,
                                        channels=3)

    with tf.Session() as sess:

        for ind, file in enumerate(f_dataset):

            # IMAGES
            image_data = (ndimage.imread(file).astype(float))
            image_data = np.transpose(image_data, [1, 0, 2])
            # Resize
            image_data = vgg_preprocessing.preprocess_image(image=image_data,
                                                            output_height=224,
                                                            output_width=224,
                                                            is_training=is_training)

            image_data = sess.run(image_data)
            np_dataset[ind, :, :, :] = image_data

            # LABELS
            classes = np.zeros(shape=[101])
            classes[int(f_labels[ind])] = 1
            np_labels[ind, :] = classes

            if ind % 50 == 0 and ind != 0:
                print('Reading and resizing images ... Step: %d' % ind)

    save = {
        'dataset': np_dataset,
        'labels': np_labels
    }

    save_pickle(pickle_file=pickle_dir, save=save)


def load_dataset(batch_size, pickle_dir):

    dataset, labels = load_pickle(pickle_dir)

    try:
        dataset = dataset[0:batch_size]
        labels  = labels[0:batch_size]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(dataset), '. ' + e)

    #return the shuffle dataset
    dataset, labels = randomize(dataset, labels)

    labels = labels.astype(np.int32)
    dataset = dataset.astype(np.float32)

    return dataset, labels

def load_train_dataset(batch_size,
                       split_dir='datasets/ucf101_splits/train_rgb_split.txt',
                       work_dir='../work/ucf101_jpegs_256/jpegs_256/',
                       is_training=True):

    train_folders, labels_folders = read_train_dataset(split_dir, work_dir)

    print('Train frames (before resize): ', len(train_folders))
    print('Train labels (before resize): ', len(labels_folders))

    try:
        train_folders = train_folders[0:batch_size]
        labels_folders = labels_folders[0:batch_size]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(train_folders), '. ' + e)

    # Allocation
    train_dataset, train_labels = make_arrays(len(train_folders),
                                              image_width=224,
                                              image_height=224,
                                              channels=3)

    labels = create_ucf101_labels()
    offset = 0

    with tf.Session() as sess:

        for ind, file in enumerate(train_folders):

            #IMAGES
            image_data = (ndimage.imread(file).astype(float))
            image_data = np.transpose(image_data, [1, 0, 2])
            #Resize
            image_data = vgg_preprocessing.preprocess_image(image=image_data,
                                                            output_height=224,
                                                            output_width=224,
                                                            is_training=is_training)


            image_data = sess.run(image_data)
            # image_data = imresize(image_data, [224, 224])
            train_dataset[ind, :, :] = image_data

            #LABELS
            classes = np.zeros(shape=[101])
            for key, value in labels.items():
                if value == labels_folders[ind]:
                    classes[key - 1] = 1

            train_labels[ind, :] = classes

            if ind % 50 == 0 and ind != 0:
                print('Reading and resizing images ... Step: %d' % ind)
                start = offset
                end = (offset + 50)

                offset = ind

                batch_images = train_dataset[start:end]
                batch_labels = train_labels[start:end]

                save = {
                    'train_dataset': batch_images,
                    'train_labels': batch_labels
                }
                pickle_file = '../work/train/ucf101_rgb_' + str(ind) + '.pickle'
                save_pickle(pickle_file=pickle_file, save=save)


    #return the shuffle dataset
    images, labels = randomize(train_dataset, train_labels)

    labels = labels.astype(np.int32)
    images = images.astype(np.float32)

    return images, labels


# train_dataset, train_labels = load_train_dataset(batch_size=9500)
#
# save = {
#     'train_dataset': train_dataset,
#     'train_labels': train_labels
# }
#
# save_pickle(pickle_file='../work/train/ucf101_rgb.pickle', save=save)

# labels = create_ucf101_labels()
#
# for key, value in labels.items():
#
#     path_folder = '../work/ucf101_jpegs_256/jpegs_256/'
#
#     # dir_folder = path_folder + value + '/'
#     # if os.path.isdir(dir_folder):
#     #     shutil.rmtree(dir_folder)
#
#     folder_label = os.path.join(path_folder, '*' + value + '*')
#     contain_label = glob.glob(folder_label)
#     print(value, len(contain_label))
#
#     for folder in contain_label:
#         path = shutil.copytree(folder, os.path.join(path_folder, value))
#         print(path)






    # flow_x_file = os.path.join(vid_name, 'flow_u_frame{0:06d}.jpg'.format(i * step + j + 1 + start_frame))
# hmdb_rgb_dataset_filename = '../work/hmdb51_jpegs_256/jpegs_256/'
# ucf_rgb_dataset_filename = '../work/ucf101_jpegs_256/jpegs_256/'
#
# hmdb_of_u_dataset_filename = '../work/hmdb51_tvl1_flow/tvl1_flow/u/'
# hmdb_of_v_dataset_filename = '../work/hmdb51_tvl1_flow/tvl1_flow/v/'
#
# ucf_of_u_dataset_filename = '../work/ucf101_tvl1_flow/tvl1_flow/u/'
# ucf_of_v_dataset_filename = '../work/ucf101_tvl1_flow/tvl1_flow/v/'
#
# dataset_folders = maybe_extract(hmdb_rgb_dataset_filename)
# train_datasets = create_pickle(dataset_folders)
#
# # dataset = 'ucf101'
# dataset = 'hmdb51'
# cmd = 'cp ../work/' + dataset + '_jpegs_256/jpegs_256/*.pickle ../work/' + dataset + '_rgb_pickles/'
#
# # cmd = 'cp ../work/' + dataset + '_tvl1_flow/tvl1_flow/u/*.pickle ../work/' + dataset + '_flow_pickles/u/'
# # cmd = 'cp ../work/' + dataset + '_tvl1_flow/tvl1_flow/v/*.pickle ../work/' + dataset + '_flow_pickles/v/'
# os.system(cmd)
# # test_folders = maybe_extract(test_filename)

# list = glob.glob('../work/hmdb51_tvl1_flow/tvl1_flow/u/*.pickle')
#
# for file in list:
#     file = file.split('/')[-1]
#     print(file)
#
#     if os.path.isfile(os.path.join('../work/hmdb51_flow_pickles/u/', file)):
#         os.remove(os.path.join('../work/hmdb51_tvl1_flow/tvl1_flow/u/', file))
#     else:
#         shutil.copyfile(os.path.join('../work/hmdb51_tvl1_flow/tvl1_flow/u/', file),
#                         os.path.join('../work/hmdb51_flow_pickles/u/', file))
#
#         os.remove(os.path.join('../work/hmdb51_tvl1_flow/tvl1_flow/u/',file))
#         print('Copy and removing file: ', file)
#
#
# print(len(glob.glob('../work/hmdb51_flow_pickles/u/*')), len(glob.glob('../work/hmdb51_tvl1_flow/tvl1_flow/u/*')))