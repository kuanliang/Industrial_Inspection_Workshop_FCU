import numpy as np
import os
import cv2
import glob
import matplotlib.image as mpimg
from .Transform import get_mask_seg

IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 1
IMAGE_SHAPE = (180, 180)
PIXEL_DEPTH = 256
path_to_images = './Data/Dataset-DetectNet_20161128_512-20170313T074149Z-001/Dataset-DetectNet_20161128_512/train/images/'


def load_images(path_to_images, shape=(256, 256), filename_index=True):
    '''load image from specified directory

    Args:
        path_to_images (string):
        shape (tuple):

    return:
        dataset (numpy 3d array):

    Notes:

    '''
    # get image paths
    image_files = [x for x in os.listdir(path_to_images) if (os.path.splitext(x)[1] == '.bmp') or
                                                            (os.path.splitext(x)[1] == '.png')]
    dataset = np.ndarray(shape=(len(image_files),
                                shape[0],
                                shape[1]),
                         dtype=np.float32)

    # fill in images into numpy array (tensor)
    file_ext = os.path.splitext(image_files[0])[1]

    # print(file_ext)
    for index, image_path in enumerate(glob.glob(path_to_images + '*{}'.format(file_ext))):
        # print(image_path)
        rgb_image = cv2.imread(image_path)
        # print(rgb_image.shape)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        image_data = (gray_image.astype(float) - PIXEL_DEPTH) / PIXEL_DEPTH

        if not filename_index:
            dataset[index, :, :] = image_data
        else:
            file_index = int(os.path.splitext(os.path.basename(image_path))[0]) - 1
            # print(file_index)
            dataset[file_index, :, :] = image_data
            # print(file_index)
    return dataset


def load_coordinates(path_to_coor):
    '''
    '''

    coord_dict = {}
    coord_dict_all = {}
    with open(path_to_coor) as f:
        coordinates = f.read().split('\n')
        for coord in coordinates:
            # print(len(coord.split('\t')))
            if len(coord.split('\t')) == 6:
                coord_dict = {}
                coord_split = coord.split('\t')
                # print(coord_split)
                # print('\n')
                coord_dict['major_axis'] = round(float(coord_split[1]))
                coord_dict['minor_axis'] = round(float(coord_split[2]))
                coord_dict['angle'] = float(coord_split[3])
                coord_dict['x'] = round(float(coord_split[4]))
                coord_dict['y'] = round(float(coord_split[5]))
                index = int(coord_split[0]) - 1
                coord_dict_all[index] = coord_dict

    return coord_dict_all


def batch_generator(path_to_images, batch_size, is_training, augment, resize):
    """Genetate training/testing image

    Args:
        path_to_image (string):
        batch_size (int):
        is_training (boolean):
        augment (boolean):

    Return:
        yield batch imagese
    """

    # image_num = len([x for x in os.listdir(path_to_images) if x.endswith('bmp')])
    image_names = [x for x in os.listdir(path_to_images) if x.endswith('bmp')]
    image_num = len(image_names)
    batch_images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    batch_masks = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    while True:
        i = 0
        for image_index in np.random.permutation(image_num):
            if is_training:
                if augment:
                    image = augment(path_to_images, image_names[image_index])
                else:
                    image = mpimg.imread(os.path.join(path_to_images, image_names[image_index]))

                mask = get_mask_seg(os.path.join(path_to_images, image_names[image_index]), xml=True)
                if resize:
                    image = cv2.resize(image, (resize[0], resize[1]))
                    mask = cv2.resize(mask, (resize[0], resize[1]))
                batch_images[i] = np.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
                batch_masks[i] = np.reshape(mask, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
            else:
                image = mpimg.imread(os.path.join(path_to_images, image_names[image_index]))
                mask = get_mask_seg(os.path.join(path_to_images, image_names[image_index]), xml=True)
                if resize:
                    image = cv2.resize(image, (resize[0], resize[1]))
                    mask = cv2.resize(mask, (resize[0], resize[1]))
                batch_images[i] = np.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
                batch_masks[i] = np.reshape(mask, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
            i += 1
            if i == batch_size:
                break

        yield batch_images, batch_masks
