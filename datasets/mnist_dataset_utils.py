from __future__ import print_function
import numpy as np
import os, sys, tarfile, shutil, glob
from scipy import ndimage
from scipy.misc import imresize
from six.moves import cPickle as pickle

np.random.seed(133)


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

# print(create_ucf101_labels())

def read_train_dataset():

    with open('datasets/ucf101_splits/train_rgb_split1.txt') as file:
        lines = file.readlines()

    train_folders = []
    labels_folders = []
    for line in lines:

        folder, frame, _ = line.split(' ')
        labels_folders.append(folder.split('/')[-2])
        folder = folder.split('/')[-1]

        # dir = os.path.join('../work/ucf101_jpegs_256/jpegs_256/', folder)
        # list_frames = os.listdir(dir)
        #
        # print(dir)
        #
        # if len(list_frames)<int(frame):
        #     frame =str(int(frame)%len(list_frames))
        #     print(frame)

        train_folders.append(os.path.join('../work/ucf101_jpegs_256/jpegs_256/', folder,'frame{0:06d}.jpg'.format(int(frame))))

    # print(train_folders)
    # print(len(lines))
    # print(labels)
    #
    with open('datasets/ucf101_splits/train_rgb_split1_1.txt', 'w+') as f:
        f.writelines(["%s\n" % item  for item in train_folders])

    return train_folders, labels_folders

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation,:]

    return shuffled_dataset, shuffled_labels

def load_train_dataset(pathfile):

    train_folders, labels_folders = read_train_dataset()

    train_folders = train_folders[0:10]
    labels_folders = labels_folders[0:10]

    # Allocation
    train_dataset, train_labels = make_arrays(len(train_folders),
                                              image_width=224,
                                              image_height=224,
                                              channels=3)

    labels = create_ucf101_labels()

    for ind, file in enumerate(train_folders):

        #IMAGES
        image_data = (ndimage.imread(file).astype(float))
        image_data = np.transpose(image_data, [1, 0, 2])
        image_data = imresize(image_data, [224, 224])
        train_dataset[ind, :, :] = image_data

        #LABELS
        classes = np.zeros(shape=[101])
        for key, value in labels.items():
            if value == labels_folders[ind]:
                classes[key - 1] = 1

        train_labels[ind, :] = classes

    return randomize(train_dataset, train_labels)


dataset, labels = load_train_dataset('')
print(dataset.shape, labels.shape)
# train_dataset = create_pickle(train_folders)






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