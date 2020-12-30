import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
from scipy.misc import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray
from frechet_inception_distance import calculate_fid_given_paths
from sewar.full_ref import uqi
from sewar.full_ref import vifp

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def evaluate(path_true, path_pred, inception_dir = '../inception'):

    psnr = []
    ssim = []
    mae = []
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

        # plt.subplot('121')
        # plt.imshow(img_gt)
        # plt.title('Groud truth')
        # plt.subplot('122')
        # plt.imshow(img_pred)
        # plt.title('Output')
        # plt.show()

        psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
        ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51))
        mae.append(compare_mae(img_gt, img_pred))
        UQI.append(uqi(img_gt, img_pred))
        VIF.append(vifp(img_gt, img_pred))
        if np.mod(index, 100) == 0:
            print(
                str(index) + ' images processed',
                "PSNR: %.4f" % round(np.mean(psnr), 4),
                "SSIM: %.4f" % round(np.mean(ssim), 4),
                "MAE: %.4f" % round(np.mean(mae), 4),
                "UQI: %.4f" % round(np.mean(UQI), 4),
                "VIF: %.4f" % round(np.mean(VIF), 4)
            )
        index += 1

    psnr = np.mean(psnr)
    mae = np.mean(mae)
    ssim = np.mean(ssim)
    UQI = np.mean(UQI)
    VIF = np.mean(VIF)
    # print(
    #     "PSNR: %.4f" % round(psnr, 4),
    #     "PSNR Variance: %.4f" % round(np.var(psnr), 4),
    #     "SSIM: %.4f" % round(ssim, 4),
    #     "SSIM Variance: %.4f" % round(np.var(ssim), 4),
    #     "MAE: %.4f" % round(mae, 4),
    #     "MAE Variance: %.4f" % round(np.var(mae), 4)
    # )

    fid_value = calculate_fid_given_paths([path_true, path_pred], '../inception')    # inception dir = '../inception'
    return mae, psnr, ssim, fid_value, UQI, VIF