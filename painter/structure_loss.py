import tensorflow as tf
import numpy as np
from skimage import feature
from skimage.color import rgb2gray

"""
Structure loss
"""
import cv2


def gaussian_kernel_2d_opencv(kernel_size=3,sigma=0):
    """
    ref: https://blog.csdn.net/qq_16013649/article/details/78784791
    ref: tensorflow
        (1) https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
        (2) https://github.com/tensorflow/tensorflow/issues/2826
    """
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky))


def canny_edge(images, sigma=1.5):
    """
    Extract edges in tensorflow.
    example:
    input = tf.placeholder(dtype=tf.float32, shape=[None, 900, 900, 3])
    output = tf.py_func(canny_edge, [input], tf.float32, stateful=False)

    :param images:
    :param sigma:
    :return:
    """
    edges = []
    for i in range(len(images)):
        grey_img = rgb2gray(images[i])
        edge = feature.canny(grey_img, sigma=sigma)
        edges.append(np.expand_dims(edge, axis=0))
    edges = np.concatenate(edges, axis=0)
    return np.expand_dims(edges, axis=3).astype(np.float32)


def priority_loss_mask(mask, ksize=5, sigma=1, iteration=2):
    gaussian_kernel = gaussian_kernel_2d_opencv(kernel_size=ksize, sigma=sigma)
    gaussian_kernel = np.reshape(gaussian_kernel, (ksize, ksize, 1, 1))
    mask_priority = tf.convert_to_tensor(mask, dtype=tf.float32)
    for i in range(iteration):
        mask_priority = tf.nn.conv2d(mask_priority, gaussian_kernel, strides=[1,1,1,1], padding='SAME')

    return mask_priority


def pyramid_structure_loss(image, predicts, edge_alpha, grad_alpha):
    _, H, W, _ = image.get_shape().as_list()
    loss = 0.
    for predict in predicts:
        _, h, w, _ = predict.get_shape().as_list()
        if h != H:
            gt_img = tf.image.resize_nearest_neighbor(image, size=(h, w))
            # gt_mask = tf.image.resize_nearest_neighbor(mask, size=(h, w))

            # grad
            gt_grad = tf.image.sobel_edges(gt_img)
            gt_grad = tf.reshape(gt_grad, [-1, h, w, 6])    # 6 channel
            grad_error = tf.abs(predict - gt_grad)

            # edge
            gt_edge = tf.py_func(canny_edge, [gt_img], tf.float32, stateful=False)
            edge_priority = priority_loss_mask(gt_edge, ksize=5, sigma=1, iteration=2)
        else:
            gt_img = image
            # gt_mask = mask

            # grad
            gt_grad = tf.image.sobel_edges(gt_img)
            gt_grad = tf.reshape(gt_grad, [-1, H, W, 6])  # 6 channel
            grad_error = tf.abs(predict - gt_grad)

            # edge
            gt_edge = tf.py_func(canny_edge, [gt_img], tf.float32, stateful=False)
            edge_priority = priority_loss_mask(gt_edge, ksize=5, sigma=1, iteration=2)

        grad_loss = tf.reduce_mean(grad_alpha * grad_error)
        edge_weight = edge_alpha * edge_priority
        # print("edge_weight", edge_weight.shape)
        # print("grad_error", grad_error.shape)
        edge_loss = tf.reduce_sum(edge_weight * grad_error) / tf.reduce_sum(edge_weight) / 6.    # 6 channel

        loss = loss + grad_loss + edge_loss

    return loss