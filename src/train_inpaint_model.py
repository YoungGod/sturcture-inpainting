import os
import time
import logging
import tensorflow as tf
import numpy as np
import time
import re

from inpaint_model import InpaintModel
from config import Config, select_gpu
from utils_fn import (show_all_variables, load_mask, create_mask,
                      save_images, load_validation_data, load_validation_mask, create_validation_mask,
                      dataset_len, load_img_scale_edge, load_val_img_scale_edge)

# Reproducible result
np.random.seed(0)
tf.set_random_seed(0)


def multi_gpu_setting(model, args):
    gpu_num = args.NUM_GPUS
    batch_size = args.BATCH_SIZE

    with tf.device("/cpu:0"):
        """ Input Data (images and masks) """
        # images and edges
        if args.CUSTOM_DATASET:
            images_edges = load_img_scale_edge(args)
        else:
            images_edges = tf.placeholder(tf.float32,
                                          [args.BATCH_SIZE * gpu_num, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]],
                                          name='real_images')
        images_, edges_, edges_128_, edges_64_ = images_edges    # a tuple
        images_ = tf.reshape(images_,
                             [args.BATCH_SIZE * gpu_num, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]])
        edges_ = tf.reshape(edges_, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        edges_128_ = tf.reshape(edges_128_, [-1, 128, 128, 1])
        edges_64_ = tf.reshape(edges_64_, [-1, 64, 64, 1])

        # masks
        if args.MASK_MODE == 'irregular':
            masks = load_mask(args)
        else:
            masks = tf.placeholder(tf.float32, [1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1],
                                   name='regular_masks')
        _masks = tf.reshape(masks, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])

        # opt
        g_optimizer = tf.train.AdamOptimizer(learning_rate=args.G_LR, beta1=0., beta2=0.9)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=args.D_LR, beta1=0., beta2=0.9)

        # update grad
        tower_g_grads = []
        tower_d_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(gpu_num):  # GPU IDs
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        _images = images_[i * batch_size: (i + 1) * batch_size]
                        _edges = edges_[i * batch_size: (i + 1) * batch_size]
                        _edges_128 = edges_128_[i * batch_size: (i + 1) * batch_size]
                        _edges_64 = edges_64_[i * batch_size: (i + 1) * batch_size]
                        print(_images.shape)
                        print(_masks.shape)
                        print(_edges.shape)
                        print(_edges_64)
                        model.build_graph_with_losses(_images, _masks, _edges, _edges_128, _edges_64, args, reuse=tf.AUTO_REUSE)
                        tf.get_variable_scope().reuse_variables()
                        # scale 256
                        _g256_grads = g_optimizer.compute_gradients(model.g_loss, var_list=model.total_g_vars)
                        _d256_grads = d_optimizer.compute_gradients(model.d_loss, var_list=model.total_d_vars)
                        tower_g_grads.append(_g256_grads)
                        with open("tower_{}_g.txt".format(i), 'w') as f:
                            for g in tower_g_grads[0]:
                                f.write("g:"+str(g)+'\n')
                        tower_d_grads.append(_d256_grads)
                        with open("tower_{}_d.txt".format(i), 'w') as f:
                            for g in tower_g_grads[0]:
                                f.write("d:"+str(g)+'\n')


        # average grads
        g_grads = average_gradients(tower_g_grads)
        d_grads = average_gradients(tower_d_grads)

        # train op
        g_train_op = g_optimizer.apply_gradients(g_grads)
        d_train_op = d_optimizer.apply_gradients(d_grads)

        # summary model in the last gpu device
        all_sum_256 = model.all_sum             # only keep the final summary

        # return train ops and inputs
        return g_train_op, d_train_op, images_edges, masks, all_sum_256


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def single_gpu_setting(model, args):
    gpu_num = args.NUM_GPUS
    assert(gpu_num == 1)

    """ Input Data (images and masks) """
    # images and edges
    if args.CUSTOM_DATASET:
        images_edges = load_img_scale_edge(args)
    else:
        images_edges = tf.placeholder(tf.float32,
                                      [args.BATCH_SIZE, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]],
                                      name='real_images')
    images_, edges_, edges_128_, edges_64_ = images_edges  # a tuple
    images = tf.reshape(images_,
                         [args.BATCH_SIZE * gpu_num, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]])
    edges = tf.reshape(edges_, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
    edges_128 = tf.reshape(edges_128_, [-1, 128, 128, 1])
    edges_64 = tf.reshape(edges_64_, [-1, 64, 64, 1])

    # masks
    if args.MASK_MODE == 'irregular':
        masks = load_mask(args)
    else:
        masks = tf.placeholder(tf.float32, [1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1],
                               name='regular_masks')
    masks = tf.reshape(masks, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])

    # build model with losses
    model.build_graph_with_losses(images, masks, edges, edges_128, edges_64, args, reuse=False)

    # train op
    g_train_op = tf.train.AdamOptimizer(learning_rate=args.G_LR, beta1=0., beta2=0.9).minimize(
        model.g_loss, var_list=model.total_g_vars)
    d_train_op = tf.train.AdamOptimizer(learning_rate=args.D_LR, beta1=0., beta2=0.9).minimize(
        model.d_loss, var_list=model.total_d_vars)

    # summary
    all_sum_256 = model.all_sum

    # return train ops and inputs
    return g_train_op, d_train_op, images_edges, masks, all_sum_256


def main():
    """
    Training
    """
    # Load config file for run an inpainting model
    args = Config('inpaint_config.yml')

    # GPU config
    gpu_ids = args.GPU_ID
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in gpu_ids])  # default "2"
    # os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu()
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True  # allow memory grow
    config_gpu.allow_soft_placement = True
    # config_gpu.log_device_placement = True

    # log setting
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("YOUNG")
    logger.setLevel(level=logging.INFO)

    """ Build Inpaint Model with Loss and Optimizer"""
    # Model and training setting
    model = InpaintModel(args)
    if args.NUM_GPUS > 1 or len(args.GPU_ID) > 1:  # multi-gpu
        logger.info("Build Inpaint Model with Loss and Optimizer in Multi-GPU setting.")
        g_train256_op, d_train256_op, images_edges, masks, all_sum_256 = multi_gpu_setting(model, args)
    else:  # cpu or single gpu
        logger.info("Build Inpaint Model with Loss and Optimizer in Single-GPU or CPU setting.")
        g_train256_op, d_train256_op, images_edges, masks, all_sum_256 = single_gpu_setting(model, args)

    # If validation?
    if args.VAL:
        logger.info("Build Validation Model.")
        with tf.device('/cpu:0'):
            # images
            images_edges_val, img_iterator_val = load_val_img_scale_edge(args)
            # masks
            if args.MASK_MODE == 'irregular':
                masks_val, mask_iterator_val = load_validation_mask(args)
            else:
                masks_val = tf.placeholder(tf.float32, [args.VAL_NUM, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1],
                                           name='val_regular_masks')
            model.build_validation_model(images_edges_val, masks_val, args)

    """ Train Logic"""
    with tf.Session(config=config_gpu) as sess:

        # Model dir
        # If restore a specific model
        if args.MODEL_RESTORE == '':
            args.MODEL_DIR =  '-'.join(time.asctime().split()) + "_GPU" + '-'.join([str(gpu) for gpu in gpu_ids]) + \
                        "_" + args.DATASET + "_" + args.GAN_TYPE + \
                        '_' + str(args.GAN_LOSS_TYPE) + str(args.PATCH_GAN_ALPHA) + \
                        "_" + "L1" + str(args.L1_FORE_ALPHA) + "_" + str(args.L1_BACK_ALPHA) + \
                        "_" + "C" + str(args.CONTENT_FORE_ALPHA) + "_" + "S" + str(args.STYLE_FORE_ALPHA) +\
                        "_" + "T" + str(args.TV_ALPHA) + "_" + args.PADDING + '_Deep_MT' +\
                        "_" + str(args.ALPHA)
        else:
            args.MODEL_DIR = args.MODEL_RESTORE

        # Checkpoint dir
        checkpoint_dir = os.path.join(args.CHECKPOINT_DIR, args.MODEL_DIR)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Sample dir
        sample_dir = os.path.join(args.SAMPLE_DIR, args.MODEL_DIR)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        # Summary writer
        writer = tf.summary.FileWriter(args.LOG_DIR + '/' + args.MODEL_DIR, sess.graph)

        # Saver to save model: to save variables
        # TODO: we can choose variables to store and steps to keep (max_to_keep)
        saver = tf.train.Saver()

        # Initialize all the variables
        tf.global_variables_initializer().run()
        # Show network architecture
        show_all_variables()

        # Try to restore model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # get checkpoint and restore training
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
            counter = 0
            logger.info(" [*] Failed to find a checkpoint")

        # Parameters
        imgh = args.IMG_SHAPES[0]
        imgw = args.IMG_SHAPES[1]

        max_step = dataset_len(args) // (args.BATCH_SIZE * args.NUM_GPUS)  # max step for each epoch
        last_step = int(args.EPOCH * max_step)             # total steps
        max_iter = last_step * args.BATCH_SIZE * args.NUM_GPUS  # max iteration when batch size is 1

        # continue to train
        if counter < last_step:
            current_epoch = counter // max_step
            current_step = counter % max_step + 1        # TODO: may not right here?
            logger.info("Start Training...")
            logger.info(
                "Total Epoch {}, Iteration per Epoch {}, Max Iteration {}, Max Iteration (batch_size=1) {}.".format(
                    args.EPOCH, max_step, last_step, max_iter))
            logger.info("Epoch Start {} at step {}".format(current_epoch, current_step))

        # not continue to train
        else:
            current_step = 0
            current_epoch = args.EPOCH

        count = 1 + counter
        for epoch in range(current_epoch, args.EPOCH):
            logger.info("Epoch {}:".format(epoch))
            time_start = time.time()
            time_s = time_start
            for step in range(current_step, max_step+1):

                # save
                if count % args.SAVE_FREQ == 0 or count == last_step:
                    saver.save(sess, os.path.join(checkpoint_dir, model.model_name + '.model'), global_step=count,write_meta_graph=False)

                if args.MASK_MODE == 'irregular':
                    # logs
                    if count % args.LOG_FREQ == 0 or count == last_step:
                        all_sum = sess.run(model.all_sum)
                        writer.add_summary(all_sum, count)
                    # train step
                    sess.run([d_train256_op, g_train256_op])
                else:
                    mask = create_mask(imgw, imgh, imgw // 2, imgh // 2, delta=0)   # random block with hole size (imgw // 2, imgh // 2)
                    # logs
                    if count % args.LOG_FREQ == 0 or count == last_step:
                        all_sum = sess.run(model.all_sum, feed_dict={masks: mask})
                        writer.add_summary(all_sum, count)
                    # train step
                    sess.run([d_train256_op, g_train256_op], feed_dict={masks: mask})

                # validation
                if args.VAL:
                    if count % args.VAL_FREQ == 0 or count == last_step:
                        sess.run(img_iterator_val.initializer)

                        if args.MASK_MODE == 'irregular':
                            sess.run(mask_iterator_val.initializer)
                            try:
                                val_all_sum = sess.run(model.val_all_sum_256)

                                writer.add_summary(val_all_sum, count)
                            except tf.errors.OutOfRangeError:
                                break
                        else:
                            try:
                                if args.STATIC_VIEW:
                                    mask = create_validation_mask(imgw, imgh, imgw // 2, imgh // 2, args, imgw // 4, imgh // 4)
                                else:
                                    mask = create_validation_mask(imgw, imgh, imgw // 2, imgh // 2, args, delta=0)
                                val_all_sum = sess.run(model.val_all_sum_256, feed_dict={masks_val: mask})

                                writer.add_summary(val_all_sum, count)
                            except tf.errors.OutOfRangeError:
                                break

                # logger info
                if count % args.PRINT_FREQ == 0 or count == last_step:
                    time_cost = (time.time() - time_start) / args.PRINT_FREQ
                    time_remaining = (last_step - count) * time_cost / 3600.
                    logger.info('epoch {}/{}, step {}/{}, cost {:.2f}s, remaining {:.2f}h.'.format(epoch, args.EPOCH, step, max_step, time_cost,time_remaining))
                    time_start = time.time()

                current_step = 0
                count += 1

            logger.info('epoch {}/{}, cost {:.2f}min.'.format(epoch, args.EPOCH, (time.time() - time_s)/60))

        logger.info("Finish.")



if __name__ == "__main__":

    main()
