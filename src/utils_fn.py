import os
import glob
import numpy as np
import scipy
from scipy.misc import imread
from scipy import ndimage
from scipy.misc import imresize

import skimage
from skimage import feature
from skimage.color import rgb2gray

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch

import cv2
# free form mask (generated by algorithm)
def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def free_form_mask_tf(parts, maxVertex=16, maxLength=60, maxBrushWidth=14, maxAngle=360, im_size=(256, 256), name='fmask'):
    """
    Free form mask
    rf: NIPS multi-column conv
    """
    # mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    with tf.variable_scope(name):
        mask = tf.Variable(tf.zeros([1, im_size[0], im_size[1], 1]), name='free_mask')
        maxVertex = tf.constant(maxVertex, dtype=tf.int32)
        maxLength = tf.constant(maxLength, dtype=tf.int32)
        maxBrushWidth = tf.constant(maxBrushWidth, dtype=tf.int32)
        maxAngle = tf.constant(maxAngle, dtype=tf.int32)
        h = tf.constant(im_size[0], dtype=tf.int32)
        w = tf.constant(im_size[1], dtype=tf.int32)
        for i in range(parts):
            p = tf.py_func(np_free_form_mask, [maxVertex, maxLength, maxBrushWidth, maxAngle, h, w], tf.float32)
            p = tf.reshape(p, [1, im_size[0], im_size[1], 1])
            mask = mask + p
        mask = tf.minimum(mask, 1.0)
    return mask

def free_form_mask(parts, maxVertex=16, maxLength=60, maxBrushWidth=14, maxAngle=360, im_size=(256, 256)):
    h, w = im_size[0], im_size[1]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    for i in range(parts):
        p = np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w)
        p = np.reshape(p, [1, h, w, 1])
        mask = mask + p
    mask = np.minimum(mask, 1.0)
    return mask

class ImageData:

    def __init__(self, args=None):
        """
        image size
        """
        self.img_size = args.IMG_SHAPES[0]
        self.channels = args.IMG_SHAPES[2]
        self.sigma = args.SIGMA
        # self.level = args.DOWN_LEVEL
        self.mode = 'rect'

    # TODO: different images with different preprocessing method
    def image_processing(self, filename):
        """
        """
        x = tf.read_file(filename,mode='RGB')    # read filename
        img = tf.image.decode_jpeg(x, channels=self.channels)    # read image and decode it. tf.image.decode_image
        img = tf.image.resize_images(img, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1    # scale to [-1, 1]
        return img

    def image_processing2(self, filename):
        img = imread(filename,mode='RGB')
        imgh, imgw = img.shape[0:2]
        if imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

            img = scipy.misc.imresize(img, [self.img_size, self.img_size])
        img = scipy.misc.imresize(img, [self.img_size, self.img_size])
        img = img.astype(np.float32) / 127.5 - 1  # scale to [-1, 1]
        return img

    def image_edge_processing(self, filename):
        img = imread(filename,mode='RGB')
        imgh, imgw = img.shape[0:2]
        if imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

            img = scipy.misc.imresize(img, [self.img_size, self.img_size])
        img = scipy.misc.imresize(img, [self.img_size, self.img_size])

        # edge
        img_gray = rgb2gray(img)  # with the channel dimension removed
        edge = feature.canny(img_gray, sigma=self.sigma).astype(np.float32)

        img = img.astype(np.float32) / 127.5 - 1  # scale to [-1, 1]
        return img, edge

    def image_edge_scale_processing(self, filename):
        img = imread(filename,mode='RGB')
        imgh, imgw = img.shape[0:2]
        if imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

            img = scipy.misc.imresize(img, [self.img_size, self.img_size])
        img = scipy.misc.imresize(img, [self.img_size, self.img_size])

        # edge
        img_gray = rgb2gray(img)  # with the channel dimension removed
        edge_256 = feature.canny(img_gray, sigma=self.sigma).astype(np.float32)
        img_gray = rgb2gray(imresize(img, [128, 128], interp='nearest'))
        edge_128 = feature.canny(img_gray, sigma=self.sigma).astype(np.float32)
        img_gray = rgb2gray(imresize(img, [64, 64], interp='nearest'))
        edge_64 = feature.canny(img_gray, sigma=self.sigma).astype(np.float32)
        # img_gray = rgb2gray(imresize(img, [32, 32]), interp='nearest')
        # edge_32 = feature.canny(img_gray, sigma=self.sigma).astype(np.float32)

        img = img.astype(np.float32) / 127.5 - 1  # scale to [-1, 1]
        return img, edge_256, edge_128, edge_64

    def mask_processing(self, filename):
        x = tf.read_file(filename)    # read mask filename
        mask = tf.image.decode_png(x, channels=1)    # read image and decode it. tf.image.decode_image
        mask = tf.image.resize_images(mask, [self.img_size, self.img_size])
        return mask

    def mask_processing2(self, filename):
        """
        For training
        """
        mask = imread(filename)

        # mask: hole = 1, data augmentation
        # mask = (mask > 0).astype(np.float32)
        # print(mask.max())
        # print(mask.min())
        mask[mask <= 127] = 0
        mask[mask > 127] = 1

        # print(mask.max())
        # print(mask.min())
        # resize
        #mask = scipy.misc.imresize(mask, (self.img_size, self.img_size))

        # random dilation (25%), we augmentation the mask in external way
        if np.random.randint(0, 4) == 0:
            mask = ndimage.binary_dilation(mask, iterations=np.random.randint(1,6)).astype(np.float32)
        mask = mask[np.newaxis, :, :, np.newaxis]

        # 5% prob generate fixed mask
        if np.random.randint(0, 20) == 0:
            mask = create_mask(256, 256, 256 // 2, 256 // 2, delta=0)

        # 10% prob generate free-form mask (ref: 2018NIPS-multi-column)
        if np.random.randint(0, 10) == 0:
            mask = free_form_mask(parts=8, im_size=(self.img_size, self.img_size),
                                     maxBrushWidth=20, maxLength=80, maxVertex=16)
        return mask.astype(np.float32)

    def mask_processing3(self, filename):
        """
        For validation and test
        """
        mask = imread(filename)
        # mask = skimage.io.imread(filename)

        # mask: hole = 1
        # mask = (mask > 0).astype(np.float32)
        mask[mask <= 127] = 0
        mask[mask > 127] = 1

        # resize
        # mask = scipy.misc.imresize(mask, (self.img_size, self.img_size))

        mask = mask[np.newaxis, :, :, np.newaxis]

        return mask.astype(np.float32)


def load_data(args):
    """
    Load image data
    """
    # training data: 0, as file list
    # image files
    with open(args.DATA_FLIST[args.DATASET][0]) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (images and masks)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)     # a tf dataset object (op)
    if args.NUM_GPUS == 1:
        device = '/gpu:0'                  # to which gpu. prefetch_to_device(device, batch_size)
    else:
        device = '/cpu:0'
    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (images and masks)
    Image_Data_Class = ImageData(args=args)

    # inputs = inputs.apply(shuffle_and_repeat(dataset_num)).apply(
    #     map_and_batch(Image_Data_Class.image_processing, args.BATCH_SIZE, num_parallel_batches=16,
    #                   drop_remainder=True)).apply(prefetch_to_device(gpu_device, args.BATCH_SIZE))
    inputs = inputs.apply(shuffle_and_repeat(dataset_num)).map(lambda filename: tf.py_func(Image_Data_Class.image_processing2, [filename], [tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(args.BATCH_SIZE*args.NUM_GPUS, drop_remainder=True).apply(prefetch_to_device(device, args.BATCH_SIZE))
    inputs_iterator = inputs.make_one_shot_iterator()  # iterator, 一次访问新的数据集的一个元素(batch)

    images = inputs_iterator.get_next()  # an iteration get a batch of data

    return images

def load_mask(args):
    # mask files
    with open(args.TRAIN_MASK_FLIST) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (masks)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)     # a tf dataset object (op)

    if args.NUM_GPUS == 1:
        device = '/gpu:0'                  # to which gpu. prefetch_to_device(device, batch_size)
    else:
        device = '/cpu:0'

    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (masks)
    Image_Data_Class = ImageData(args=args)

    # inputs = inputs.apply(shuffle_and_repeat(dataset_num)).apply(
    #     map_and_batch(Image_Data_Class.image_processing, args.BATCH_SIZE, num_parallel_batches=16,
    #                   drop_remainder=True)).apply(prefetch_to_device(gpu_device, args.BATCH_SIZE))
    inputs = inputs.apply(shuffle_and_repeat(dataset_num)).map(lambda filename: tf.py_func(
        Image_Data_Class.mask_processing2, [filename], [tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(1,drop_remainder=True).apply(prefetch_to_device(device, 1))
    # inputs = inputs.apply(prefetch_to_device(device))
    inputs_iterator = inputs.make_one_shot_iterator()  # iterator

    masks = inputs_iterator.get_next()  # an iteration get a batch of data

    return masks

def create_mask(width, height, mask_width, mask_height, x=None, y=None, delta=0):
    """
    create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)
    delta: margin between mask and image boundary
    """
    mask = np.zeros((height, width))
    mask_x = x if x is not None else np.random.randint(delta, width - mask_width - delta)
    mask_y = y if y is not None else np.random.randint(delta, height - mask_height - delta)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    mask = mask[np.newaxis, :, :, np.newaxis]
    return mask

def load_validation_data(args):
    """
    Load image data
    """
    # validation data: 1, as file list
    # image files
    with open(args.DATA_FLIST[args.DATASET][1]) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (images)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)     # a tf dataset object (op)

    gpu_device = '/gpu:0'                  # to which gpu. prefetch_to_device(device, batch_size)
    # gpu_device = '/gpu:{}'.format(args.GPU_ID)

    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (images)
    Image_Data_Class = ImageData(args)
    inputs = inputs.map(lambda filename: tf.py_func(Image_Data_Class.image_processing2, [filename], [tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(args.VAL_NUM,drop_remainder=True).apply(prefetch_to_device(gpu_device,1))
    inputs_iterator = inputs.make_initializable_iterator()  # iterator, need to be initialized

    images = inputs_iterator.get_next()  # an iteration get a batch of data

    return images, inputs_iterator

def load_validation_mask(args):
    # mask files
    with open(args.VAL_MASK_FLIST) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (masks)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)     # a tf dataset object (op)

    gpu_device = '/gpu:0'                  # to which gpu. prefetch_to_device(device, batch_size)
    # gpu_device = '/gpu:{}'.format(args.GPU_ID)

    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (masks)
    Image_Data_Class = ImageData(args=args)

    inputs = inputs.map(lambda filename: tf.py_func(Image_Data_Class.mask_processing3, [filename], [tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(args.VAL_NUM,drop_remainder=True).apply(prefetch_to_device(gpu_device, 1))
    # inputs = inputs.apply(prefetch_to_device(gpu_device))
    inputs_iterator = inputs.make_initializable_iterator()

    masks = inputs_iterator.get_next()  # an iteration get a batch of data

    return masks, inputs_iterator

def create_validation_mask(width, height, mask_width, mask_height, args, x=None, y=None, delta=0):
    """
    create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)
    """
    masks = np.zeros((args.VAL_NUM, height, width))
    for i in range(args.VAL_NUM):
        mask_x = x if x is not None else np.random.randint(delta, width - mask_width - delta)
        mask_y = y if y is not None else np.random.randint(delta, height - mask_height - delta)
        masks[i,mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    masks = masks[:, :, :, np.newaxis]
    return masks

def load_test_data(args):
    """
    Load image data
    """
    # test data: 2, as file list
    # image files
    with open(args.DATA_FLIST[args.DATASET][1]) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (images)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)     # a tf dataset object (op)

    gpu_device = '/gpu:0'                  # to which gpu. prefetch_to_device(device, batch_size)
    # gpu_device = '/gpu:{}'.format(args.GPU_ID)

    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (images)
    Image_Data_Class = ImageData(args=args)
    inputs = inputs.map(lambda filename: tf.py_func(Image_Data_Class.image_processing2, [filename], [tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(args.TEST_NUM,drop_remainder=True).apply(prefetch_to_device(gpu_device))
    inputs_iterator = inputs.make_initializable_iterator()  # iterator, need to be initialized

    images = inputs_iterator.get_next()  # an iteration get a batch of data

    return images, inputs_iterator

def load_test_mask(args):
    # mask files
    with open(args.TEST_MASK_FLIST) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (masks)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)     # a tf dataset object (op)

    gpu_device = '/gpu:0'                  # to which gpu. prefetch_to_device(device, batch_size)
    # gpu_device = '/gpu:{}'.format(args.GPU_ID)

    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (masks)
    Image_Data_Class = ImageData(args=args)

    inputs = inputs.map(lambda filename: tf.py_func(Image_Data_Class.mask_processing3, [filename], [tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(args.TEST_NUM,drop_remainder=True).apply(prefetch_to_device(gpu_device, 1))
    # inputs = inputs.apply(prefetch_to_device(gpu_device))
    inputs_iterator = inputs.make_initializable_iterator()  # iterator

    masks = inputs_iterator.get_next()  # an iteration get a batch of data

    return masks, inputs_iterator

def create_test_mask(width, height, mask_width, mask_height, args, x=None, y=None, delta=0):
    """
    create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)
    """
    masks = np.zeros((args.TEST_NUM, height, width))
    for i in range(args.TEST_NUM):
        mask_x = x if x is not None else np.random.randint(delta, width - mask_width - delta)
        mask_y = y if y is not None else np.random.randint(delta, height - mask_height - delta)
        masks[i,mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    masks = masks[:, :, :, np.newaxis]
    return masks

def dataset_len(args):
    with open(args.DATA_FLIST[args.DATASET][0]) as f:
        fnames = f.read().splitlines()
    return len(fnames)

def show_all_variables():
    """
    Show all the variables of an tf model.
    """
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)*127.5

def load_img_edge(args):
    """
    Load image data
    """
    # training data: 0, as file list
    # image files
    with open(args.DATA_FLIST[args.DATASET][0]) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (images and masks)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)  # a tf dataset object (op)
    if args.NUM_GPUS == 1:
        device = '/gpu:0'  # to which gpu. prefetch_to_device(device, batch_size)
        # gpu_device = '/gpu:{}'.format(args.GPU_ID)
    else:
        device = '/cpu:0'
    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (images and masks)
    Image_Data_Class = ImageData(args=args)
    # inputs = inputs.apply(shuffle_and_repeat(dataset_num)).apply(
    #     map_and_batch(Image_Data_Class.image_processing, args.BATCH_SIZE, num_parallel_batches=16,
    #                   drop_remainder=True)).apply(prefetch_to_device(gpu_device, args.BATCH_SIZE))
    inputs = inputs.apply(shuffle_and_repeat(dataset_num)).map(lambda filename: tf.py_func(
        Image_Data_Class.image_edge_processing, [filename], [tf.float32, tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(args.BATCH_SIZE*args.NUM_GPUS, drop_remainder=True).apply(prefetch_to_device(device, args.BATCH_SIZE))
    inputs_iterator = inputs.make_one_shot_iterator()  # iterator

    images_edges = inputs_iterator.get_next()  # an iteration get a batch of data

    return images_edges

def load_val_img_edge(args):

    """
    Load image data
    """
    # validation data: 1, as file list
    # image files
    with open(args.DATA_FLIST[args.DATASET][1]) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (images)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)  # a tf dataset object (op)

    gpu_device = '/gpu:0'  # to which gpu. prefetch_to_device(device, batch_size)
    # gpu_device = '/gpu:{}'.format(args.GPU_ID)

    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (images)
    Image_Data_Class = ImageData(args)
    inputs = inputs.map(lambda filename: tf.py_func(Image_Data_Class.image_edge_processing, [filename], [tf.float32, tf.float32]),
                        num_parallel_calls=3)
    inputs = inputs.batch(args.VAL_NUM, drop_remainder=True).apply(prefetch_to_device(gpu_device))
    inputs_iterator = inputs.make_initializable_iterator()  # iterator, need to be initialized

    images_edges = inputs_iterator.get_next()  # an iteration get a batch of data

    return images_edges, inputs_iterator

def load_test_img_edge(args):

    """
    Load image data
    """
    # validation data: 1, as file list
    # image files
    with open(args.DATA_FLIST[args.DATASET][1]) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (images)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)  # a tf dataset object (op)

    gpu_device = '/gpu:0'  # to which gpu. prefetch_to_device(device, batch_size)
    # gpu_device = '/gpu:{}'.format(args.GPU_ID)

    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (images)
    Image_Data_Class = ImageData(args)
    inputs = inputs.map(lambda filename: tf.py_func(Image_Data_Class.image_edge_processing, [filename], [tf.float32, tf.float32]),
                        num_parallel_calls=3)
    inputs = inputs.batch(args.TEST_NUM, drop_remainder=True).apply(prefetch_to_device(gpu_device))
    inputs_iterator = inputs.make_initializable_iterator()  # iterator, need to be initialized

    images_edges = inputs_iterator.get_next()  # an iteration get a batch of data

    return images_edges, inputs_iterator

def load_img_scale_edge(args):
    """
    Load image data
    """
    # training data: 0, as file list
    # image files
    with open(args.DATA_FLIST[args.DATASET][0]) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (images and masks)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)  # a tf dataset object (op)
    if args.NUM_GPUS == 1:
        device = '/gpu:0'  # to which gpu. prefetch_to_device(device, batch_size)
        # gpu_device = '/gpu:{}'.format(args.GPU_ID)
    else:
        device = '/cpu:0'
    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (images and masks)
    Image_Data_Class = ImageData(args=args)

    # inputs = inputs.apply(shuffle_and_repeat(dataset_num)).apply(
    #     map_and_batch(Image_Data_Class.image_processing, args.BATCH_SIZE, num_parallel_batches=16,
    #                   drop_remainder=True)).apply(prefetch_to_device(gpu_device, args.BATCH_SIZE))
    inputs = inputs.apply(shuffle_and_repeat(dataset_num)).map(lambda filename: tf.py_func(
        Image_Data_Class.image_edge_scale_processing, [filename], [tf.float32, tf.float32,tf.float32, tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(args.BATCH_SIZE*args.NUM_GPUS, drop_remainder=True).apply(prefetch_to_device(device, args.BATCH_SIZE*args.NUM_GPUS))
    inputs_iterator = inputs.make_one_shot_iterator()  # iterator

    images_edges = inputs_iterator.get_next()  # an iteration get a batch of data

    return images_edges

def load_val_img_scale_edge(args):
    """
    Load image data
    """
    # training data: 0, as file list
    # image files
    with open(args.DATA_FLIST[args.DATASET][1]) as f:
        fnames = f.read().splitlines()

    # TODO: create input dataset (images and masks)
    inputs = tf.data.Dataset.from_tensor_slices(fnames)     # a tf dataset object (op)

    gpu_device = '/gpu:0'                  # to which gpu. prefetch_to_device(device, batch_size)
    # gpu_device = '/gpu:{}'.format(args.GPU_ID)

    dataset_num = len(fnames)
    # TODO: dataset with preprocessing (images and masks)
    Image_Data_Class = ImageData(args=args)

    # inputs = inputs.apply(shuffle_and_repeat(dataset_num)).apply(
    #     map_and_batch(Image_Data_Class.image_processing, args.BATCH_SIZE, num_parallel_batches=16,
    #                   drop_remainder=True)).apply(prefetch_to_device(gpu_device, args.BATCH_SIZE))
    inputs = inputs.apply(shuffle_and_repeat(dataset_num)).map(lambda filename: tf.py_func(
        Image_Data_Class.image_edge_scale_processing, [filename], [tf.float32, tf.float32,tf.float32, tf.float32]), num_parallel_calls=3)
    inputs = inputs.batch(args.VAL_NUM,drop_remainder=True).apply(prefetch_to_device(gpu_device, args.BATCH_SIZE))
    inputs_iterator = inputs.make_initializable_iterator()  # iterator

    images_edges = inputs_iterator.get_next()  # an iteration get a batch of data

    return images_edges, inputs_iterator



# random rect mask
def random_bbox(config):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.img_shapes
    img_height = img_shape[0]
    img_width = img_shape[1]
    if config.random_mask is True:
        maxt = img_height - config.margins[0] - config.mask_shapes[0]
        maxl = img_width - config.margins[1] - config.mask_shapes[1]
        t = tf.random_uniform(
            [], minval=config.margins[0], maxval=maxt, dtype=tf.int32)
        l = tf.random_uniform(
            [], minval=config.margins[1], maxval=maxl, dtype=tf.int32)
    else:
        t = config.mask_shapes[0]//2
        l = config.mask_shapes[1]//2
    h = tf.constant(config.mask_shapes[0])
    w = tf.constant(config.mask_shapes[1])
    return (t, l, h, w)


def bbox2mask(bbox, config, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.img_shapes
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             config.max_delta_shapes[0], config.max_delta_shapes[1]],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask

"""
How to use 
# generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=8, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=20, maxLength=80, maxVertex=16)
"""