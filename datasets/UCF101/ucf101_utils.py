from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys
from scipy import ndimage
from scipy.misc import imresize
from six.moves import cPickle as pickle
import scipy.io as sio
import cv2
import random
import glob
import math

np.random.seed(133)

def create_ucf101_labels():
    with open('datasets/ucf101_splits/classInd.txt') as file:
        lines = file.readlines()

    labels = {}

    for line in lines:
        label = line.split(' ')
        labels[int(label[0])-1] = label[1].split('\n')[0]

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

    print('Split dimensions: ', len(dataset), len(labels))

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

def load_dataset_from_split(batch_size, split_dir, offset=0):

    dataset, labels = read_split(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        f_labels = labels[offset:(offset + batch_size)]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    np_dataset, np_labels = make_arrays(len(f_dataset),
                                        image_width=224,
                                        image_height=224,
                                        channels=3)

    for ind, file in enumerate(f_dataset):

        # IMAGES
        image_data = (ndimage.imread(file).astype(float))
        image_data = np.transpose(image_data, [1, 0, 2])
        # RESIZE
        image_data = imresize(image_data, [224, 224])
        np_dataset[ind, :, :, :] = image_data

        # LABELS
        classes = np.zeros(shape=[101])
        classes[int(f_labels[ind])] = 1
        np_labels[ind, :] = classes

        if ind % 50 == 0 and ind != 0:
            print('Reading and resizing images ... Step: %d' % ind)

    #return the shuffle dataset
    np_dataset, np_labels = randomize(np_dataset, np_labels)

    np_dataset = np_dataset.astype(np.float32)
    np_labels = np_labels.astype(np.int32)

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    return np_dataset, np_labels

def generate_pickle_dataset(batch_size, split_dir, pickle_dir, offset=0, is_training=False):

    dataset, labels = read_split(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        f_labels  = labels[offset:(offset + batch_size)]

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
            image_data = datasets.vgg_preprocessing.preprocess_image(image=image_data,
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

def read_txt(split_dir):

    dataset = []
    max_frames = []
    labels = []

    with open(split_dir) as file:
        lines = file.readlines()

        for line in lines:
            folder_path = line.split(' ')[0].split('/')[-1]
            max_num_frames = line.split(' ')[1]
            label = line.split(' ')[2].split('\n')[0]

            dataset.append(folder_path)
            max_frames.append(max_num_frames)
            labels.append(label)

    print('Split dimensions: ', len(dataset), len(max_frames), len(labels))

    return dataset, max_frames, labels

def preparare_txt_datasets():
    folders, frames, labels = read_txt('datasets/ucf101_splits/ucf101_rgb_train.txt')

    new_file = []

    folders = folders[::]
    counter = 0
    for ind, folder in enumerate(folders):

        max_num_frames = int(frames[ind]) - 1

        for frame in np.random.randint(low=1, high=max_num_frames, size=5):
            frame_path = os.path.join(folder, 'frame{0:06d}.jpg'.format(frame))
            new_file.append(frame_path + ' ' + labels[ind] + '\n')
            counter += 1

    _frames = []
    for frame in frames:
        _frames.append(int(frame) - 1)
    print(min(_frames), max(_frames))
    print('Counter:', counter)
    print(len(folders))
    print(len(frames))
    print(len(labels))
    print(len(new_file))

    random.shuffle(new_file)

    with open('datasets/train_datasets/ucf101_rgb_split1.txt', 'wr') as f:
        f.writelines(new_file[:20000])

    with open('datasets/train_datasets/ucf101_rgb_split2.txt', 'wr') as f:
        f.writelines(new_file[20000:40000])

    with open('datasets/train_datasets/ucf101_rgb_split3.txt', 'wr') as f:
        f.writelines(new_file[40000:60000])

    with open('datasets/train_datasets/ucf101_rgb_valid.txt', 'wr') as f:
        f.writelines(new_file[60000::])

def load_dataset_data_augmentation(batch_size, split_dir, offset=0):

    dataset, labels = read_split(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        f_labels = labels[offset:(offset + batch_size)]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    #Memory allocation
    dataset_size = 10 * len(f_dataset)
    np_dataset, np_labels = make_arrays(dataset_size,
                                        image_width=224,
                                        image_height=224,
                                        channels=3)

    d = sio.loadmat('datasets/UCF101/image_mean/rgb_mean.mat')
    image_mean = d['image_mean']

    dims = (256, 340, 3)

    for ind, file in enumerate(f_dataset):

        rgb = np.zeros(shape=dims, dtype=np.float64)
        rgb_flip = np.zeros(shape=dims, dtype=np.float64)
        classes = np.zeros(shape=[101])

        start = ind * 10
        end = start + 10

        # LABELS
        print(f_dataset[ind])
        print(f_labels[ind])
        classes[int(f_labels[ind])] = 1
        np_labels[start:end, :] = classes

        # IMAGES
        rgb_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        # RESIZE
        rgb_image = cv2.resize(rgb_image, dims[1::-1])

        rgb[:, :, :] = rgb_image
        rgb_flip[:, :, :] = rgb_image[:, ::-1, :]

        # CROP
        rgb_1 = rgb[:224, :224, :]
        rgb_2 = rgb[:224, -224:, :]
        rgb_3 = rgb[16:240, 60:284, :]
        rgb_4 = rgb[-224:, :224, :]
        rgb_5 = rgb[-224:, -224:, :]
        rgb_f_1 = rgb_flip[:224, :224, :]
        rgb_f_2 = rgb_flip[:224, -224:, :]
        rgb_f_3 = rgb_flip[16:240, 60:284, :]
        rgb_f_4 = rgb_flip[-224:, :224, :]
        rgb_f_5 = rgb_flip[-224:, -224:, :]

        rgb_1 = np.expand_dims(rgb_1, axis=3)
        rgb_2 = np.expand_dims(rgb_2, axis=3)
        rgb_3 = np.expand_dims(rgb_3, axis=3)
        rgb_4 = np.expand_dims(rgb_4, axis=3)
        rgb_5 = np.expand_dims(rgb_5, axis=3)

        rgb_f_1 = np.expand_dims(rgb_f_1, axis=3)
        rgb_f_2 = np.expand_dims(rgb_f_2, axis=3)
        rgb_f_3 = np.expand_dims(rgb_f_3, axis=3)
        rgb_f_4 = np.expand_dims(rgb_f_4, axis=3)
        rgb_f_5 = np.expand_dims(rgb_f_5, axis=3)

        rgb = np.concatenate((rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_f_1, rgb_f_2, rgb_f_3, rgb_f_4, rgb_f_5), axis=3)

        # SUBSTRACT MEAN
        rgb = rgb[...] - np.tile(image_mean[..., np.newaxis], (1, 1, 1, rgb.shape[3]))

        rgb = np.transpose(rgb, [3, 1, 0, 2])

        np_dataset[start:end, : , :, :] = rgb

        if ind % 100 == 0 and ind != 0:
            print('Reading and resizing images ... Step: %d' % ind)

    # return the shuffle dataset
    # np_dataset, np_labels = randomize(np_dataset, np_labels)

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    # np_dataset = np_dataset.astype(np.float32)
    # np_labels = np_labels.astype(np.int32)

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    return np_dataset, np_labels

def rgb_data_augmentation(img_input, dims):

    d = sio.loadmat('datasets/UCF101/image_mean/rgb_mean.mat')
    image_mean = d['image_mean']

    image = np.zeros(shape=dims, dtype=np.float32)
    image_flip = np.zeros(shape=dims, dtype=np.float32)


    image[:, :, :] = img_input
    image_flip[:, :, :] = img_input[:, ::-1, :]

    # CROP
    image_1 = image[:224, :224, :]
    image_2 = image[:224, -224:, :]
    image_3 = image[16:240, 60:284, :]
    image_4 = image[-224:, :224, :]
    image_5 = image[-224:, -224:, :]
    image_f_1 = image_flip[:224, :224, :]
    image_f_2 = image_flip[:224, -224:, :]
    image_f_3 = image_flip[16:240, 60:284, :]
    image_f_4 = image_flip[-224:, :224, :]
    image_f_5 = image_flip[-224:, -224:, :]

    image_1 = np.expand_dims(image_1, axis=3)
    image_2 = np.expand_dims(image_2, axis=3)
    image_3 = np.expand_dims(image_3, axis=3)
    image_4 = np.expand_dims(image_4, axis=3)
    image_5 = np.expand_dims(image_5, axis=3)

    image_f_1 = np.expand_dims(image_f_1, axis=3)
    image_f_2 = np.expand_dims(image_f_2, axis=3)
    image_f_3 = np.expand_dims(image_f_3, axis=3)
    image_f_4 = np.expand_dims(image_f_4, axis=3)
    image_f_5 = np.expand_dims(image_f_5, axis=3)

    image = np.concatenate((image_1, image_2, image_3, image_4, image_5, image_f_1, image_f_2, image_f_3, image_f_4, image_f_5), axis=3)

    # SUBSTRACT MEAN
    image = image[...] - np.tile(image_mean[..., np.newaxis], (1, 1, 1, image.shape[3]))

    image = np.transpose(image, [3, 1, 0, 2])

    return image

def load_train_rgb_dataset(batch_size, offset, split_dir, work_dir):

    dataset, labels = read_split(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        f_labels = labels[offset:(offset + batch_size)]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    #Memory allocation
    dims = (256, 340, 3)
    dataset_size = len(f_dataset) * 10
    np_dataset, np_labels = make_arrays(dataset_size,
                                        image_width=224,
                                        image_height=224,
                                        channels=3)

    for ind, file in enumerate(f_dataset):

        classes = np.zeros(shape=[101], dtype=np.int32)
        start = ind * 10
        end = start + 10

        # LABELS
        classes[int(f_labels[ind])] = 1
        np_labels[start:end, :] = classes

        # IMAGES
        file_path = os.path.join(work_dir, file)
        rgb_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        # RESIZE
        rgb_image = cv2.resize(rgb_image, dims[1::-1])

        rgb = rgb_data_augmentation(img_input=rgb_image, dims=dims)

        np_dataset[start:end, :, :, :] = rgb

        if ind % 100 == 0 and ind != 0:
            print('(Read + resize + data augmentation) over images ... Step: %d' % ind)

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    return np_dataset, np_labels

def flow_data_augmentation(img_input, flip_img_input, dims):

    d = sio.loadmat('datasets/UCF101/image_mean/flow_mean.mat')
    image_mean = d['image_mean']

    flow = np.zeros(shape=dims, dtype=np.float32)
    flow_flip = np.zeros(shape=dims, dtype=np.float32)


    flow[:, :, :, :] = img_input
    flow_flip[:, :, :, :] = flip_img_input

    # CROP
    flow_1 = flow[:224, :224, :, :]
    flow_2 = flow[:224, -224:, :, :]
    flow_3 = flow[16:240, 60:284, :, :]
    flow_4 = flow[-224:, :224, :, :]
    flow_5 = flow[-224:, -224:, :, :]
    flow_f_1 = flow_flip[:224, :224, :, :]
    flow_f_2 = flow_flip[:224, -224:, :, :]
    flow_f_3 = flow_flip[16:240, 60:284, :, :]
    flow_f_4 = flow_flip[-224:, :224, :, :]
    flow_f_5 = flow_flip[-224:, -224:, :, :]

    flow = np.concatenate((flow_1, flow_2, flow_3, flow_4, flow_5, flow_f_1, flow_f_2, flow_f_3, flow_f_4, flow_f_5), axis=3)

    # SUBSTRACT MEAN
    flow = flow[...] - np.tile(image_mean[..., np.newaxis], (1, 1, 1, flow.shape[3]))

    flow = np.transpose(flow, [3, 1, 0, 2])

    return flow

def spatial_data_augmentation(img_input, flip_img_input, dims):

    d = sio.loadmat('datasets/UCF101/image_mean/rgb_mean.mat')
    image_mean = d['image_mean']

    rgb = np.zeros(shape=dims, dtype=np.float32)
    rgb_flip = np.zeros(shape=dims, dtype=np.float32)


    rgb[:, :, :, :] = img_input
    rgb_flip[:, :, :, :] = flip_img_input

    # CROP
    rgb_1 = rgb[:224, :224, :, :]
    rgb_2 = rgb[:224, -224:, :, :]
    rgb_3 = rgb[16:240, 60:284, :, :]
    rgb_4 = rgb[-224:, :224, :, :]
    rgb_5 = rgb[-224:, -224:, :, :]
    rgb_f_1 = rgb_flip[:224, :224, :, :]
    rgb_f_2 = rgb_flip[:224, -224:, :, :]
    rgb_f_3 = rgb_flip[16:240, 60:284, :, :]
    rgb_f_4 = rgb_flip[-224:, :224, :, :]
    rgb_f_5 = rgb_flip[-224:, -224:, :, :]

    rgb = np.concatenate((rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_f_1, rgb_f_2, rgb_f_3, rgb_f_4, rgb_f_5), axis=3)

    # SUBSTRACT MEAN
    rgb = rgb[...] - np.tile(image_mean[..., np.newaxis], (1, 1, 1, rgb.shape[3]))

    rgb = np.transpose(rgb, [3, 1, 0, 2])

    return rgb

def load_validation_flow_dataset(batch_size, offset, split_dir, work_dir, num_samples):

    dataset, duration, labels = read_txt(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        duration = duration[offset:(offset + batch_size)]
        f_labels = labels[offset:(offset + batch_size)]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    # Memory allocation
    optical_flow_frames = 10
    start_frame = 0

    dims = (256, 340, optical_flow_frames * 2, num_samples)

    dataset_size = len(f_dataset) * (num_samples * 10)

    np_dataset, np_labels = make_arrays(dataset_size,
                                        image_width=224,
                                        image_height=224,
                                        channels=20)

    for ind, file in enumerate(f_dataset):

        classes = np.zeros(shape=[101], dtype=np.int32)
        start = ind * (num_samples * 10)
        end = start + (num_samples * 10)

        # LABELS
        classes[int(f_labels[ind])] = 1
        np_labels[start:end, :] = classes

        max_frames = int(duration[ind])

        step = int(math.floor((max_frames - optical_flow_frames + 1) / num_samples))
        flow = np.zeros(shape=dims, dtype=np.float32)
        flow_flip = np.zeros(shape=dims, dtype=np.float32)

        for i in range(num_samples):

            for j in range(optical_flow_frames):

                # IMAGES
                flow_x_file = os.path.join(work_dir, 'u', file, 'frame{0:06d}.jpg'.format(i * step + j + 1 + start_frame))
                flow_y_file = os.path.join(work_dir, 'v', file, 'frame{0:06d}.jpg'.format(i * step + j + 1 + start_frame))

                img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)

                img_x = cv2.resize(img_x, dims[1::-1])
                img_y = cv2.resize(img_y, dims[1::-1])

                # flow.shape [256, 340, 20, 20]
                flow[:, :, j * 2, i] = img_x
                flow[:, :, j * 2 + 1, i] = img_y

                # flow.shape [256, 340, 20, 20]
                flow_flip[:, :, j * 2, i] = 255 - img_x[:, ::-1]
                flow_flip[:, :, j * 2 + 1, i] = img_y[:, ::-1]

        # flow.shape [200, 224, 224, 20]
        flow = flow_data_augmentation(img_input=flow, flip_img_input=flow_flip, dims=dims)

        np_dataset[start:end, :, :, :] = flow

        if ind % 100 == 0 and ind != 0:
            print('(Read + resize + data augmentation) over images ... Step: %d' % ind)

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    return np_dataset, np_labels

def load_validation_spatial_dataset(batch_size, offset, split_dir, work_dir, num_samples):

    dataset, duration, labels = read_txt(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        f_duration = duration[offset:(offset + batch_size)]
        f_labels = labels[offset:(offset + batch_size)]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    # Memory allocation
    dataset_size = len(f_dataset) * (num_samples * 10)

    np_dataset, np_labels = make_arrays(dataset_size,
                                        image_width=224,
                                        image_height=224,
                                        channels=3)

    for ind, file in enumerate(f_dataset):

        classes = np.zeros(shape=[101], dtype=np.int32)
        start = ind * (num_samples * 10)
        end = start + (num_samples * 10)

        # LABELS
        classes[int(f_labels[ind])] = 1
        np_labels[start:end, :] = classes

        max_frames = int(f_duration[ind])

        step = int(math.floor((max_frames - 1) / (num_samples - 1)))

        dims = (256, 340, 3, num_samples)

        rgb = np.zeros(shape=dims, dtype=np.float32)
        rgb_flip = np.zeros(shape=dims, dtype=np.float32)

        for i in range(num_samples):

            # IMAGES
            img_file = os.path.join(work_dir, file, 'frame{0:06d}.jpg'.format(i * step + 1))

            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dims[1::-1])

            # rgb.shape, rgb_flip.shape [256, 340, 3, 20]
            rgb[:, :, :, i] = img
            rgb_flip[:, :, :, i] = img[:, ::-1, :]

        # rgb.shape [200, 224, 224, 3]
        rgb = spatial_data_augmentation(img_input=rgb, flip_img_input=rgb_flip, dims=dims)

        # [X:X+200, 224, 224 , 3]
        np_dataset[start:end, :, :, :] = rgb

        if ind % 100 == 0 and ind != 0:
            print('(Read + resize + data augmentation) over images ... Step: %d' % ind)

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    return np_dataset, np_labels

def load_train_dataset(batch_size, offset, split_dir, work_dir, mode):

    if mode == 'RGB':
        return load_train_rgb_dataset(batch_size, offset, split_dir, work_dir)
    elif mode == 'FLOW':
        return load_train_flow_dataset(batch_size, offset, split_dir, work_dir)
    else:
        raise Exception('Mode train dataset not supported.')

def load_validation_rgb_dataset(batch_size, offset, split_dir, work_dir):

    dataset, duration, labels = read_txt(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        f_duration = duration[offset:(offset + batch_size)]
        f_labels = labels[offset:(offset + batch_size)]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    dims = (256, 340, 3)
    num_samples = 20
    start_frame = 0

    dataset_size = len(f_dataset) * 200

    np_dataset, np_labels = make_arrays(dataset_size,
                                        image_width=224,
                                        image_height=224,
                                        channels=3)

    for i, file in enumerate(f_dataset):

        classes = np.zeros(shape=[101], dtype=np.int32)

        start = i * (num_samples * 10)
        end = start + (num_samples * 10)

        # LABELS
        classes[int(f_labels[i])] = 1
        np_labels[start:end, :] = classes

        max_frames = int(f_duration[i])

        step = int(math.floor((max_frames - 1) / (num_samples - 1)))

        rgb_image = np.zeros(shape=dims, dtype=np.float32)
        rgb_images = np.zeros(shape=(200, 224, 224, 3), dtype=np.float32)

        for j in range(num_samples):

            f_start = j * (10)
            f_end = f_start + (10)

            # IMAGES
            img_file = os.path.join(work_dir, file, 'frame{0:06d}.jpg'.format(step * j + 1 + start_frame))

            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dims[1::-1])

            # rgb_image.shape [256, 340, 3]
            rgb_image[:, :, :] = img

            # rgb_images.shape [200, 224, 224, 3]
            rgb_images[f_start:f_end, :, :, :] = rgb_data_augmentation(img_input=rgb_image, dims=dims)


        print('(Read + resize + data augmentation) over clip %s (step: %d)' % (file, i))

        np_dataset[start:end, :, :, :] = rgb_images

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    return np_dataset, np_labels

def load_train_spatial_dataset(batch_size, offset, split_dir, work_dir, num_samples):

    dataset, duration, labels = read_txt(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        f_duration = duration[offset:(offset + batch_size)]
        f_labels = labels[offset:(offset + batch_size)]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    # Memory allocation
    dataset_size = len(f_dataset) * (num_samples)

    np_dataset, np_labels = make_arrays(dataset_size,
                                        image_width=224,
                                        image_height=224,
                                        channels=3)

    for ind, file in enumerate(f_dataset):

        # LABELS
        classes = np.zeros(shape=[101], dtype=np.int32)
        classes[int(f_labels[ind])] = 1
        np_labels[ind, :] = classes

        max_frames = int(f_duration[ind])

        dims = (256, 340, 3, num_samples)

        rgb = np.zeros(shape=dims, dtype=np.float32)
        rgb_flip = np.zeros(shape=dims, dtype=np.float32)

        for i in range(num_samples):
            # IMAGES
            img_file = os.path.join(work_dir, file, 'frame{0:06d}.jpg'.format(np.random.randint(1, max_frames - 1)))

            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dims[1::-1])

            # rgb.shape, rgb_flip.shape [256, 340, 3, 20]
            rgb[:, :, :, i] = img
            rgb_flip[:, :, :, i] = img[:, ::-1, :]

        # rgb.shape [10, 224, 224, 3]
        rgb = spatial_data_augmentation(img_input=rgb, flip_img_input=rgb_flip, dims=dims)

        delete_pos = sorted(random.sample(range(rgb.shape[0]), rgb.shape[0] - 1))
        rgb = np.delete(rgb, delete_pos, axis=0)

        # [1, 224, 224 , 3]
        np_dataset[ind, :, :, :] = rgb

        if ind % 100 == 0 and ind != 0:
            print('(Read + resize + data augmentation) over images ... Step: %d' % ind)
            # print('deleted positions: ', sorted(delete_pos))

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    return np_dataset, np_labels

def load_train_flow_dataset(batch_size, offset, split_dir, work_dir, num_samples):

    dataset, duration, labels = read_txt(split_dir)

    try:
        f_dataset = dataset[offset:(offset + batch_size)]
        duration = duration[offset:(offset + batch_size)]
        f_labels = labels[offset:(offset + batch_size)]

    except Exception as e:
        print('Batch size too big: ', batch_size, '. Length train_folder: ', len(f_dataset), '. ' + e)

    # Memory allocation
    optical_flow_frames = 10
    start_frame = 0

    dims = (256, 340, optical_flow_frames * 2, num_samples)

    dataset_size = len(f_dataset) * (num_samples)

    np_dataset, np_labels = make_arrays(dataset_size,
                                        image_width=224,
                                        image_height=224,
                                        channels=20)

    for ind, file in enumerate(f_dataset):

        classes = np.zeros(shape=[101], dtype=np.int32)
        start = ind * (num_samples)
        end = start + (num_samples)

        # LABELS
        classes[int(f_labels[ind])] = 1
        np_labels[start:end, :] = classes

        max_frames = int(duration[ind])

        start_frame = np.random.randint(0, (max_frames-(optical_flow_frames + 1)))

        step = int(math.floor((max_frames - optical_flow_frames + 1) / num_samples))
        flow = np.zeros(shape=dims, dtype=np.float32)
        flow_flip = np.zeros(shape=dims, dtype=np.float32)

        for i in range(num_samples):

            for j in range(optical_flow_frames):
                # IMAGES
                flow_x_file = os.path.join(work_dir, 'u', file,
                                           'frame{0:06d}.jpg'.format(i * step + j + 1 + start_frame))
                flow_y_file = os.path.join(work_dir, 'v', file,
                                           'frame{0:06d}.jpg'.format(i * step + j + 1 + start_frame))

                img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)

                img_x = cv2.resize(img_x, dims[1::-1])
                img_y = cv2.resize(img_y, dims[1::-1])

                # flow.shape [256, 340, 20, 1]
                flow[:, :, j * 2, i] = img_x
                flow[:, :, j * 2 + 1, i] = img_y

                # flow.shape [256, 340, 20, 1]
                flow_flip[:, :, j * 2, i] = 255 - img_x[:, ::-1]
                flow_flip[:, :, j * 2 + 1, i] = img_y[:, ::-1]

        # flow.shape [10, 224, 224, 20]
        flow = flow_data_augmentation(img_input=flow, flip_img_input=flow_flip, dims=dims)

        delete_pos = sorted(random.sample(range(flow.shape[0]), flow.shape[0] - 1))
        flow = np.delete(flow, delete_pos, axis=0)

        # flow.shape [1, 224, 224, 20]
        np_dataset[start:end, :, :, :] = flow

        if ind % 100 == 0 and ind != 0:
            print('(Read + resize + data augmentation) over images ... Step: %d' % ind)

    print('Dataset dimensions: ', np_dataset.shape, np_labels.shape)

    return np_dataset, np_labels


        # with open('datasets/UCF101/ucf101_splits/ucf101_rgb_train.txt', 'r') as f:
#     lines = f.readlines()

# full_lines = []
#
# random.shuffle(lines)

# for line in lines:
#     full_lines.append(line.split('/')[-1])
#
# with open('datasets/UCF101/ucf101_splits/testlist03.txt', 'r') as f:
#     lines = f.readlines()
#
# split_lines = []
# for line in lines:
#     for full_line in full_lines:
#         if full_line.split(' ')[0] == line.split('/')[-1].split('.')[0]:
#             split_lines.append(full_line)
#
# random.shuffle(split_lines)
# with open('datasets/train_datasets/ucf101_testlist03.txt', 'wr+') as f:
#     f.writelines(split_lines)

# # 13.320 / 3 = 4.440
# train_split1 = new_lines[0:4000]
# valid_split1 = new_lines[4000:4440]
#
# train_split2 = new_lines[4440:8440]
# valid_split2 = new_lines[8440:8880]
#
# train_split3 = new_lines[8880:12880]
# valid_split3 = new_lines[12880:13320]
#
#
# with open('datasets/train_datasets/ucf101_train_split1.txt', 'wr+') as f:
#     f.writelines(train_split1)
#
# with open('datasets/valid_datasets/ucf101_valid_split1.txt', 'wr+') as f:
#     f.writelines(valid_split1)
#
# with open('datasets/train_datasets/ucf101_train_split2.txt', 'wr+') as f:
#     f.writelines(train_split2)
#
# with open('datasets/valid_datasets/ucf101_valid_split2.txt', 'wr+') as f:
#     f.writelines(valid_split2)
#
# with open('datasets/train_datasets/ucf101_train_split3.txt', 'wr+') as f:
#     f.writelines(train_split3)
#
# with open('datasets/valid_datasets/ucf101_valid_split3.txt', 'wr+') as f:
#     f.writelines(valid_split3)

# frames = []
# for line in lines:
#     frames.append(int(line.split(' ')[1]))



# ucf_list = os.listdir('../work/ucf101_tvl1_flow/tvl1_flow/u/')
#
# labels = create_ucf101_labels()
# list_file = []
#
# with open('datasets/ucf101_flow_list.txt', 'wr+') as f:
#
#     for line in ucf_list:
#         label = line.split('_')[1]
#
#         for key, value in labels.items():
#             if value == label:
#                 num_l = int(key)
#
#         print(label, num_l)
#
#         frame_list = os.listdir('../work/ucf101_tvl1_flow/tvl1_flow/u/' + line)
#
#         list_file.append('../work/ucf101_tvl1_flow/tvl1_flow/u/' + line + ' ' + str(len(frame_list)) + ' ' + str(num_l) + '\n')
#
#     f.writelines(list_file)
#
# with open('datasets/ucf101_train_list.txt', 'wr+') as f:
#
#     for line in ucf_list:
#         label = line.split('_')[1]
#
#         for key, value in labels.items():
#             if value == label:
#                 num_l = int(key)
#
#         print(label, num_l)
#
#         frame_list = os.listdir('../work/ucf101_tvl1_flow/tvl1_flow/u/' + line)
#
#         list_file.append(line + ' ' + str(len(frame_list)) + ' ' + str(num_l) + '\n')
#
#     f.writelines(list_file)

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