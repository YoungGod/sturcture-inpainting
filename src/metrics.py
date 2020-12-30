import tensorflow as tf

def psnr(x, pred_x, max_val=255):
    """
    PSNR
    """
    val = tf.reduce_mean(tf.image.psnr(x, pred_x, max_val=max_val))
    return val

def ssmi(x, pred_x, max_val=255):
    """
    SSMI
    """
    val = tf.reduce_mean(tf.image.ssim(x, pred_x, max_val=max_val))
    return val

def mm_ssmi(x, pred_x, max_val=255):
    """
    MM-SSMI
    """
    val = tf.reduce_mean(tf.image.ssim_multiscale(x, pred_x, max_val=max_val))
    return val

def avg_l1(x, pred_x):
    val = tf.reduce_mean(tf.abs(x - pred_x))
    return val

def tv_loss(pred_x):
    N, H, W, C = pred_x.shape.as_list()
    size = H*W*C
    val = tf.reduce_mean(tf.image.total_variation(pred_x)) / size
    return val

import numpy as np
from glob import glob
from ntpath import basename
from scipy.misc import imread
from skimage.color import rgb2gray
from sewar.full_ref import uqi
from sewar.full_ref import vifp

def uqi_vif(path_true, path_pred):

    UQI = []
    VIF = []
    names = []
    index = 1

    files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
    for fn in sorted(files):
        name = basename(str(fn))
        names.append(name)

        img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
        img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)

        img_gt = rgb2gray(img_gt)
        img_pred = rgb2gray(img_pred)

        UQI.append(uqi(img_gt, img_pred))
        VIF.append(vifp(img_gt, img_pred))
        if np.mod(index, 100) == 0:
            print(
                str(index) + ' images processed',
                "UQI: %.4f" % round(np.mean(UQI), 4),
                "VIF: %.4f" % round(np.mean(VIF), 4),
            )
        index += 1

    UQI = np.mean(UQI)
    VIF = np.mean(VIF)

    return UQI, VIF