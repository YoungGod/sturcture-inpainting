import os
import time
import logging
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import re

from math import ceil
from scipy.misc import imsave
from inpaint_model import InpaintModel
from config import Config, select_gpu
from utils_fn import show_all_variables, load_test_data, load_test_mask, create_test_mask, dataset_len, load_test_img_edge

from frechet_inception_distance import calculate_fid_given_paths
from metrics import uqi_vif

# For reproducible result
np.random.seed(0)
tf.set_random_seed(0)

# with tf.device('/cpu:0'):
"""
Testing
"""
# Load config file for run an inpainting model
args = Config('inpaint_config.yml')

# GPU config
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu()
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True  # allow memory grow

# log setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YOUNG")
logger.setLevel(level=logging.INFO)

""" Input Data (images and masks) """
# images
if args.CUSTOM_DATASET:
    images, image_iterator = load_test_img_edge(args)
else:
    images = tf.placeholder(tf.float32, [args.BATCH_SIZE, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]],
                                 name='real_images')
# test masks
if args.MASK_MODE == 'irregular':
    masks, mask_iterator = load_test_mask(args)
else:
    masks = tf.placeholder(tf.float32, [args.TEST_NUM, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1],
                                 name='test_regular_masks')

""" Build Testing Inpaint Model"""
# Testing model
model = InpaintModel(args)
logger.info("Build Testing Inpaint Model")
model.build_test_model(images, masks, args)

""" Testing Logic"""
with tf.Session(config=config_gpu) as sess:

    # Saver to restore model: to restore variables
    # TODO: we can choose variables to store and steps to keep (max_to_keep)
    saver = tf.train.Saver()

    # Model dir
    # If restore a specific model
    args.MODEL_DIR = args.MODEL_RESTORE

    # Result dirs
    # (1) result/model_dir/inpainted_images
    # (2) result/model_dir/masked_images
    # (3) result/model_dir/sample_images
    result_dir = os.path.join(args.RESULT_DIR, args.MODEL_DIR)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # (1) result/model_dir/inpainted_images
    inpainted_dir = os.path.join(result_dir, 'inpainted_images')
    if not os.path.exists(inpainted_dir):
        os.makedirs(inpainted_dir)
    # (2) result/model_dir/maked_images
    masked_dir = os.path.join(result_dir, 'masked_images')
    if not os.path.exists(masked_dir):
        os.makedirs(masked_dir)
    # (3) result/model_dir/sample_images
    sample_dir = os.path.join(result_dir, 'sample_images')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    # (4) result/model_dir/masks
    mask_dir = os.path.join(result_dir, 'masks')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    # (5) result/model_dir/inpainted_smpales
    inpainted_sample_dir = os.path.join(result_dir, 'inpainted_samples')
    if not os.path.exists(inpainted_sample_dir):
        os.makedirs(inpainted_sample_dir)

    # Model Checkpoint dir
    checkpoint_dir = os.path.join(args.CHECKPOINT_DIR, args.MODEL_DIR)

    # Testing data info
    with open(args.DATA_FLIST[args.DATASET][1]) as f:
        fnames = f.read().splitlines()

    data_len = len(fnames)
    max_test_step = ceil(data_len / args.TEST_NUM)  # TEST_NUM can be 1 or a batch like 8
    max_test_step = min(max_test_step, ceil(args.MAX_TEST_NUM / args.TEST_NUM))    # max test number of images

    # Training data info
    max_step = dataset_len(args) // args.BATCH_SIZE  # max step for each epoch
    last_step = int(args.EPOCH * max_step)  # total steps
    # Parameters
    imgh = args.IMG_SHAPES[0]
    imgw = args.IMG_SHAPES[1]

    # Try to restore model
    # Initialize all the variables
    tf.global_variables_initializer().run()
    # Show network architecture
    show_all_variables()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        # print ckpt name with dir
        logger.info("Latest ckpt: {}".format(ckpt.model_checkpoint_path))
        logger.info("All ckpt: {}".format(ckpt.all_model_checkpoint_paths))
        # ckpt base name
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        # restore
        # saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))  # restore
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            try:
                var_value = tf.contrib.framework.load_variable(os.path.join(checkpoint_dir, ckpt_name), from_name)
                assign_ops.append(tf.assign(var, var_value))
            except Exception:
                continue
        sess.run(assign_ops)
        print('Model loaded.')

        counter = int(next(re.finditer("\d+", ckpt_name)).group(0))
        logger.info(" [*] Success to read {}".format(ckpt_name))
    else:
        logger.info(" [*] Failed to find a checkpoint")
    # Existing Training info
    current_epoch = counter // max_step
    current_step = counter % max_step
    logger.info('Evaluating epoch {}, step {}.'.format(current_epoch, current_step))

    # Testing start
    # For saving evaluation results
    if not os.path.exists(os.path.join(result_dir, 'evaluation.csv')):
        with open(os.path.join(result_dir, 'evaluation.csv'), mode='a') as f:
            f.write("epoch, step, l1, pnsr, ssim, fid, uqi, vif\n")
    mask_size = []
    l1_list = []
    psnr_list = []
    ssim_list = []

    count = 1
    sess.run(image_iterator.initializer)
    if args.MASK_MODE == 'irregular':
        sess.run(mask_iterator.initializer)
    for step in range(1, max_test_step+1):
        time_start = time.time()

        try:
            if args.MASK_MODE == 'irregular':
                raw_x, raw_x_incomplete, raw_x_complete, mask, l1, psnr, ssim = sess.run([model.raw_x, model.raw_x_incomplete,
                                                                                          model.raw_x_complete, model.mask,
                                                                                          model.l1, model.psnr, model.ssim])
            else:
                mask = create_test_mask(imgw, imgh, imgw // 2, imgh // 2, args)
                raw_x, raw_x_incomplete, raw_x_complete, mask, l1, psnr, ssim = sess.run([model.raw_x, model.raw_x_incomplete,
                                                                                          model.raw_x_complete, model.mask,
                                                                                          model.l1, model.psnr, model.ssim],
                                                                                         feed_dict={masks: mask})
        except tf.errors.OutOfRangeError:
            break

        # setting hole pixel value = 255
        ones_x = np.ones_like(raw_x_incomplete)
        raw_x_incomplete = raw_x_incomplete + ones_x*mask*255

        for i in range(args.TEST_NUM):
            # save result
            imsave(os.path.join(sample_dir, args.DATASET+"{}.png".format(count)), raw_x[i])
            imsave(os.path.join(inpainted_dir, args.DATASET+"{}.png".format(count)), raw_x_complete[i])
            imsave(os.path.join(masked_dir, args.DATASET+"{}.png".format(count)), raw_x_incomplete[i])
            imsave(os.path.join(mask_dir, args.DATASET+"{}.png".format(count)), mask[i, :, :, 0])    # mask is grey image

            # mask size
            mask_size.append(mask[i].sum())
            l1_list.append(l1[i])
            psnr_list.append(psnr[i])
            ssim_list.append(ssim[i])
            
            if step == 1:
                imsave(os.path.join(inpainted_sample_dir, args.DATASET + "{}_{}.png".format(count, current_epoch)), raw_x_complete[i])

            count += 1

        time_cost = time.time() - time_start
        time_remaining = (max_test_step - step) * time_cost
        logger.info(
            'step {}/{}, image {}/{}, cost {:.2f}s, remaining {:.2f}s.'.format(step, max_test_step, count, data_len, time_cost,
                                                                               time_remaining))

    # Final evaluation
    # df_evaluation = pd.DataFrame(data=np.array([l1_list, psnr_list, ssim_list, mask_size]).T,
    #                              columns=["l1", "psnr", "ssim", "mask"])
    # df_evaluation.to_csv(os.path.join(result_dir, 'evaluation.csv'), index=False)
    logger.info("Saving Finished.")

    logger.info("Evaluating Results..")
    # Evaluation result
    # l1, psnr, ssim, fid

    # fid score
    logger.info("FID score")
    fid_value = calculate_fid_given_paths([sample_dir, inpainted_dir], 'inception')
    print("FID: ", fid_value)

    # print(df_evaluation.mean(axis=0))
    # UQI and VIF
    uqi, vif = uqi_vif(sample_dir, inpainted_dir)
    # uqi, vif = 0., 0.
    # df_evaluation = pd.concat(df_evaluation, pd.DataFrame(data={"epoch": current_epoch, "step": current_step,
    #                                                             "l1": np.array(l1_list).mean(),
    #                                                             "psnr": np.array(psnr_list).mean(),
    #                                                             "ssim": np.array(ssim_list).mean(),
    #                                                             "fid": fid_value}), axis=0)
    # df_evaluation.to_csv(os.path.join(result_dir, 'evaluation.csv'), index=False)
    with open(os.path.join(result_dir, 'evaluation.csv'), mode='a') as f:
        f.write("{}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(current_epoch, current_step,
                                                                  np.array(l1_list).mean(),
                                                                  np.array(psnr_list).mean(),
                                                                  np.array(ssim_list).mean(),
                                                                  fid_value,
                                                                                  uqi,
                                                                                  vif))

    logger.info("Evaluation Finished.")