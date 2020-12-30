import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from utils_fn import *
from ops import *
from loss import *
from metrics import *


class InpaintModel():

    def __init__(self, args):
        self.model_name = "InpaintModel"   # name for checkpoint
        self.dataset_name = args.DATASET    # TODO: get the name of the data set, it depends the path structure
        self.checkpoint_dir = args.CHECKPOINT_DIR
        self.sample_dir = args.SAMPLE_DIR
        self.result_dir = args.RESULT_DIR
        self.log_dir = args.LOG_DIR

        self.epoch = args.EPOCH
        self.batch_size = args.BATCH_SIZE
        self.print_freq = args.PRINT_FREQ
        self.save_freq = args.SAVE_FREQ
        self.img_size = args.IMG_SHAPES

    # yj
    def build_inpaint_net(self, x, edge, grad, mask, args=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image[-1, 1] with shape of (batch_size, h, w, c)
            edge: incomplete edge {0, 1} with shape of (batch_size, h, w)
            grad map: incomplete grad with shape of (batch_size, h, w, 6)
            mask: mask region {0, 1}
        Returns:
            complete image, grad map, middle result
        """
        x = tf.reshape(x, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]])
        mask = tf.reshape(mask, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        edge = tf.reshape(edge, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        # grad = tf.reshape(grad, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 6])

        xin = x
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x * edge, ones_x * mask, grad], axis=3)  # add a mask channel,the input channel is 4
        # encoder-decoder network: channel 64-128-256-128-64
        cnum = 64  # initial channel
        # a decorate: arg_scope([op1, op2,..], xx,..) means:
        # attributes or parameters xx defined here are the default in op1 and op2,..
        with tf.variable_scope(name, reuse=reuse), \
             arg_scope([gen_conv, gen_deconv],
                       training=training, padding=padding):
            # Encoder
            # scale 256                                                                     channels   activation: relu
            x = gen_conv(x, cnum, 7, stride=1, activation=tf.nn.relu, name='en_conv1')  # 9 -> 64, ksize=7x7, stride=1
            # scale 128
            x = gen_conv(x, 2 * cnum, 4, stride=2, activation=tf.nn.relu, name='en_conv2')
            # scale 64
            x = gen_conv(x, 4 * cnum, 4, stride=2, activation=tf.nn.relu, name='en_conv3')
            # res block
            x = resnet_blocks(x, 4 * cnum, 3, stride=1, rate=2, block_num=8, activation=tf.nn.relu, name='en_64_8')

            # Decoder
            # TODO: output scale 64  Down scale = 2 (origin) pool scale = 2 (origin)
            # share attention
            x = attention(x, 4 * cnum, down_scale=2, pool_scale=2, name='attention_pooling_64')

            # out of predict grad map
            x_64 = gen_conv(x, 4 * cnum, 5, stride=1, activation=tf.nn.relu, name='out64_grad_out')
            x_grad_out_64 = gen_conv(x_64, 6, 1, stride=1, activation=None, name='grad64')
            x_out_64 = gen_conv(x_64, 3, 1, stride=1, activation=tf.nn.tanh, name='out64')

            # scale 64 - 128
            x = tf.concat([x, x_64], axis=3)
            x = gen_deconv(x, 2 * cnum, 4, method='deconv', activation=tf.nn.relu, name='de128_conv4_upsample')

            # TODO: output scale 128
            # share attention
            x = attention(x, 2 * cnum, down_scale=2, pool_scale=2, name='attention_pooling_128')

            # out of predict grad map
            x_128 = gen_conv(x, 2 * cnum, 5, stride=1, activation=tf.nn.relu, name='out128_grad_out')
            x_grad_out_128 = gen_conv(x_128, 6, 1, stride=1, activation=None, name='grad128')
            x_out_128 = gen_conv(x_128, 3, 1, stride=1, activation=tf.nn.tanh, name='out128')

            # scale 128 - 256
            x = tf.concat([x, x_128], axis=3)
            x = gen_deconv(x, cnum, 4, method='deconv', activation=tf.nn.relu, name='de256_conv5_upsample')

            # TODO: output scale 256
            # share attention
            x = attention(x, cnum, down_scale=2, pool_scale=2, name='attention_pooling_256')

            # out of predict grad map
            x = gen_conv(x, cnum, 5, stride=1, activation=tf.nn.relu, name='out256_grad_out')
            x_grad = gen_conv(x, 6, 1, stride=1, activation=None, name='grad256')  # grad map  no activation
            x = gen_conv(x, 3, 1, stride=1, activation=tf.nn.tanh, name='out256')

        return x, x_out_64, x_out_128, x_grad, x_grad_out_64, x_grad_out_128

    # yj
    def build_discriminator_256(self, x, reuse=False, name='discriminator256', sn=True, training=True):
        """
        Patch GAN discriminator component, receptive filed: 70*70
        """
        with tf.variable_scope(name, reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, ksize=4, stride=2, name='conv1', sn=sn, training=training)     # leaky_relu
            x = dis_conv(x, cnum*2, ksize=4, stride=2, name='conv2', sn=sn, training=training)
            x = dis_conv(x, cnum*4, ksize=4, stride=2, name='conv3', sn=sn, training=training)
            x = dis_conv(x, cnum*8, ksize=4, stride=1, name='conv4', sn=sn, training=training)
            x = dis_conv(x, 1, ksize=4, stride=1, name='conv5', activation=None, sn=sn, training=training)
            return x

    # yj
    def build_graph_with_losses(self, x, mask, edge, edge_128, edge_64, args, training=True, reuse=False):
        # Orgin image, edge, grad
        # image, edge, edge_128, edge_64 = x
        grad = tf.image.sobel_edges(x)  # normalization?
        grad = tf.reshape(grad, [args.BATCH_SIZE, 256, 256, 6])  # 6 channel

        # x for image
        # x = tf.reshape(image, [args.BATCH_SIZE, args.IMG_SHAPES[0], args.IMG_SHAPES[1],
        #                        args.IMG_SHAPES[2]])  # [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]]
        # mask = tf.reshape(mask, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        # edge = tf.reshape(edge, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        # edge_128 = tf.reshape(edge_128, [-1, 128, 128, 1])
        # edge_64 = tf.reshape(edge_64, [-1, 64, 64, 1])

        # incomplete image at full scale
        x_incomplete = x * (1. - mask)  # mask: 0 for valid pixel, 1 (white) for hole

        # incomplete edge at full scale
        input_edge = 1 - edge                 # 0 (black) for edge when save and input, 1 (white) for non edge
        edge_incomplete = input_edge * (1 - mask) + mask

        # incomplete grad
        grad_incomplete = (1. - mask) * grad

        # bulid inpaint net
        out_256, out_64, out_128, out_grad_256, out_grad_64, out_grad_128 = self.build_inpaint_net(x_incomplete,
                                                             edge_incomplete, grad_incomplete,
                                                             mask, args, reuse=reuse,
                                                             training=training, padding=args.PADDING)

        """##### Losses #####"""
        losses = {}  # use a dict to collect losses

        # TODO: scale 64
        # complete image
        mask_64 = tf.image.resize_nearest_neighbor(mask, (64, 64))
        x_pos_64 = tf.image.resize_nearest_neighbor(x, (64, 64))  # pos input (real)

        x_incomplete_64 = x_pos_64 * (1. - mask_64)
        x_complete_64 = out_64 * mask_64 + x_incomplete_64
        x_neg_64 = x_complete_64  # neg input (fake)

        # Auxilary task: edge and grad loss
        grad_64 = tf.image.sobel_edges(x_pos_64)  # normalization?
        grad_64 = tf.reshape(grad_64, [args.BATCH_SIZE, 64, 64, 6])  # 6 channel
        grad_incomplete_64 = (1. - mask_64) * grad_64
        grad_complete_64 = out_grad_64 * mask_64 + grad_incomplete_64

        # more weight for edges?
        edge_mask_64 = edge_64                                       # 1 for edge, 0 for grad, when using feature.canny()
        mask_priority_64 = priority_loss_mask(edge_mask_64, ksize=5, sigma=1, iteration=2)
        edge_weight_64 = args.EDGE_ALPHA * mask_priority_64    # salient edge

        grad_weight_64 = args.GRAD_ALPHA                       # equaled grad

        # error
        grad_error_64 = tf.abs(out_grad_64 - grad_64)

        # edge loss
        losses['edge_l1_loss_64'] = tf.reduce_sum(edge_weight_64 * grad_error_64) / tf.reduce_sum(edge_weight_64) / 6.

        # grad pixel level reconstruction loss
        if args.GRAD_ALPHA > 0:
            losses['grad_l1_loss_64'] = tf.reduce_mean(grad_weight_64 * grad_error_64)
        else:
            losses['grad_l1_loss_64'] = 0.
        # grad patch level loss with implicit nearest neighbor matching
        if args.GRAD_MATCHING_ALPHA > 0:
            losses['grad_matching_64'] = args.GRAD_MATCHING_ALPHA * grad_matching_loss(out_grad_64, grad_64, args)
        else:
            losses['grad_matching_64'] = 0.

        # Main task: compute losses
        # l1 loss
        # if args.L1_SCALE > 0.:
        loss_caculator = LossCalculator(args.VGG_DIR, x_pos_64)  # x_pose_256: real image
        losses['l1_loss_fore_64'] = args.L1_SCALE * args.L1_FORE_ALPHA * loss_caculator.l1_loss(x_pos_64, out_64, mask_64,
                                                                                 type='foreground')
        losses['l1_loss_back_64'] = args.L1_SCALE * args.L1_BACK_ALPHA * loss_caculator.l1_loss(x_pos_64, out_64, mask_64,
                                                                                 type='background')
        # else:
        #     losses['l1_loss_fore_64'] = 0.
        #     losses['l1_loss_back_64'] = 0.

        self.losses_64 = [losses['l1_loss_fore_64'],
                          losses['l1_loss_back_64'],
                          losses['grad_l1_loss_64'],
                          losses['edge_l1_loss_64'],
                          losses['grad_matching_64']
                          ]
        # Summary
        viz_img_64 = [x_pos_64, x_incomplete_64, x_complete_64]
        viz_grad_64 = [grad_64[:, :, :, 0:1], grad_incomplete_64[:, :, :, 0:1], grad_complete_64[:, :, :, 0:1]]
        self.img_64 = tf.concat(viz_img_64, axis=2)
        self.grad_64 = tf.concat(viz_grad_64, axis=2)

        all_sum_64 = [tf.summary.scalar("l1_loss_fore_64", losses['l1_loss_fore_64']),
                       tf.summary.scalar("l1_loss_back_64", losses['l1_loss_back_64']),
                       tf.summary.image('raw_incomplete_predicted_complete_64',
                                        tf.concat(viz_img_64, axis=2), max_outputs=args.VIZ_MAX_OUT),
                       tf.summary.image('raw_incomplete_predicted_completed_grad_64',
                                        tf.concat(viz_grad_64, axis=2), max_outputs=args.VIZ_MAX_OUT),
                       tf.summary.scalar('grad_l1_loss_64', losses['grad_l1_loss_64']),
                       tf.summary.scalar('edge_l1_loss_64', losses['edge_l1_loss_64']),
                       tf.summary.scalar('grad_matching_64', losses['grad_matching_64']),
                       ]

        # TODO: scale 128
        # complete image
        mask_128 = tf.image.resize_nearest_neighbor(mask,(128, 128))
        x_pos_128 = tf.image.resize_nearest_neighbor(x, (128, 128))     # pos input (real)

        x_incomplete_128 = x_pos_128 * (1. - mask_128)
        x_complete_128 = out_128 * mask_128 + x_incomplete_128
        x_neg_128 = x_complete_128                                      # neg input (fake)

        # Auxilary task: edge and grad loss
        grad_128 = tf.image.sobel_edges(x_pos_128)  # normalization?
        grad_128 = tf.reshape(grad_128, [args.BATCH_SIZE, 128, 128, 6])  # 6 channel
        grad_incomplete_128 = (1. - mask_128) * grad_128
        grad_complete_128 = out_grad_128 * mask_128 + grad_incomplete_128

        # more weight for edges?
        # edge weight
        edge_mask_128 = edge_128  # 1 for edge, 0 for grad, when using feature.canny()
        mask_priority_128 = priority_loss_mask(edge_mask_128, ksize=5, sigma=1, iteration=2)
        edge_weight_128 = args.EDGE_ALPHA * mask_priority_128  # salient edge
        # grad weight
        grad_weight_128 = args.GRAD_ALPHA  # equaled grad

        # error
        grad_error_128 = tf.abs(out_grad_128 - grad_128)

        # edge loss
        losses['edge_l1_loss_128'] = tf.reduce_sum(edge_weight_128 * grad_error_128) / tf.reduce_sum(edge_weight_128) / 6.

        # grad pixel level reconstruction loss
        if args.GRAD_ALPHA > 0:
            losses['grad_l1_loss_128'] = tf.reduce_mean(grad_weight_128 * grad_error_128)
        else:
            losses['grad_l1_loss_128'] = 0.
        # grad patch level loss with implicit nearest neighbor matching
        if args.GRAD_MATCHING_ALPHA > 0:
            losses['grad_matching_128'] = args.GRAD_MATCHING_ALPHA * grad_matching_loss(out_grad_128, grad_128, args)
        else:
            losses['grad_matching_128'] = 0.
        # Main task
        # compute losses
        # if args.L1_SCALE > 0.:
        loss_caculator = LossCalculator(args.VGG_DIR, x_pos_128)    # x_pose_256: real image

        # l1 loss
        losses['l1_loss_fore_128'] = args.L1_SCALE * args.L1_FORE_ALPHA * loss_caculator.l1_loss(x_pos_128, out_128, mask_128,
                                                                                 'foreground')
        losses['l1_loss_back_128'] = args.L1_SCALE * args.L1_BACK_ALPHA * loss_caculator.l1_loss(x_pos_128, out_128, mask_128,
                                                                                     'background')
        # else:
        #     losses['l1_loss_fore_128'] = 0.
        #     losses['l1_loss_back_128'] = 0.

        self.losses_128 = [losses['l1_loss_fore_128'],
                           losses['l1_loss_back_128'],
                           losses['grad_l1_loss_128'],
                           losses['edge_l1_loss_128'],
                           losses['grad_matching_128']
                           ]
        # Summary
        viz_img_128 = [x_pos_128, x_incomplete_128, x_complete_128]
        viz_grad_128 = [grad_128[:, :, :, 0:1], grad_incomplete_128[:, :, :, 0:1], grad_complete_128[:, :, :, 0:1]]

        self.img_128 = tf.concat(viz_img_128, axis=2)
        self.grad_128 = tf.concat(viz_grad_128, axis=2)

        all_sum_128 = [tf.summary.scalar("l1_loss_fore_128", losses['l1_loss_fore_128']),
                      tf.summary.scalar("l1_loss_back_128", losses['l1_loss_back_128']),
                      tf.summary.image('raw_incomplete_predicted_complete_128',
                                       tf.concat(viz_img_128, axis=2), max_outputs=args.VIZ_MAX_OUT),
                      tf.summary.image('raw_incomplete_predicted_completed_grad_128',
                                       tf.concat(viz_grad_128, axis=2), max_outputs=args.VIZ_MAX_OUT),
                      tf.summary.scalar('grad_l1_loss_128', losses['grad_l1_loss_128']),
                      tf.summary.scalar('edge_l1_loss_128', losses['edge_l1_loss_128']),
                      tf.summary.scalar('grad_matching_128', losses['grad_matching_128']),
                      ]

        # TODO: scale 256
        # apply mask and complete image
        mask_256 = mask
        x_incomplete_256 = x_incomplete
        x_complete_256 = out_256 * mask_256 + x_incomplete_256

        # Auxilary task: edge and grad loss
        grad_256 = grad  # normalization?
        grad_256 = tf.reshape(grad_256, [args.BATCH_SIZE, 256, 256, 6])  # 6 channel
        grad_incomplete_256 = (1. - mask_256) * grad_256
        grad_complete_256 = out_grad_256 * mask_256 + grad_incomplete_256

        # more weight for edges?
        # edge weight
        edge_mask_256 = edge  # 1 for edge, 0 for grad, when using feature.canny()
        mask_priority_256 = priority_loss_mask(edge_mask_256, ksize=5, sigma=1, iteration=2)
        edge_weight_256 = args.EDGE_ALPHA * mask_priority_256  # salient edge
        # grad weight
        grad_weight_256 = args.GRAD_ALPHA  # equaled grad

        # error
        grad_error_256 = tf.abs(out_grad_256 - grad_256)

        # edge loss
        losses['edge_l1_loss_256'] = tf.reduce_sum(edge_weight_256 * grad_error_256) / tf.reduce_sum(edge_weight_256) / 6.

        # grad pixel level reconstruction loss
        if args.GRAD_ALPHA > 0:
            losses['grad_l1_loss_256'] = tf.reduce_mean(grad_weight_256 * grad_error_256)
        else:
            losses['grad_l1_loss_256'] = 0.
        # grad patch level loss with implicit nearest neighbor matching
        if args.GRAD_MATCHING_ALPHA > 0:
            losses['grad_matching_256'] = args.GRAD_MATCHING_ALPHA * grad_matching_loss(out_grad_256, grad_256, args)
        else:
            losses['grad_matching_256'] = 0.

        # compute losses
        x_neg_256 = x_complete_256  # neg input (fake)
        x_pos_256 = x  # pos input (real)

        # losses
        loss_caculator = LossCalculator(args.VGG_DIR, x_pos_256)    # x_pose_256: real image

        # l1 loss
        losses['l1_loss_fore_256'] = args.L1_FORE_ALPHA * loss_caculator.l1_loss(x_pos_256, out_256, mask_256,
                                                                                 'foreground')
        losses['l1_loss_back_256'] = args.L1_BACK_ALPHA * loss_caculator.l1_loss(x_pos_256, out_256, mask_256,
                                                                                 'background')

        # image patch level loss with implicit nearest neighbor matching
        if args.IMG_MATCHING_ALPHA > 0:
            losses['img_matching_256'] = args.IMG_MATCHING_ALPHA * grad_matching_loss(out_256, x, args)
        else:
            losses['img_matching_256'] = 0.

        # content loss, style loss, tv loss
        layers = {'relu1_1': 0.2, 'relu2_1': 0.2, 'relu3_1': 0.2, 'relu4_1': 0.2, 'relu5_1': 0.2}
        if args.CONTENT_FORE_ALPHA > 0.:
            layers = {'relu1_1': 0.2, 'relu2_1': 0.2, 'relu3_1': 0.2, 'relu4_1': 0.2, 'relu5_1': 0.2}
            losses['content_loss_256'] = args.CONTENT_FORE_ALPHA * loss_caculator.content_loss(x_neg_256, layers=layers)
        else:
            losses['content_loss_256'] = 0.
        if args.STYLE_FORE_ALPHA > 0.:
            # layers = {'pool1': 0.33, 'pool2': 0.34, 'pool3': 0.33}
            losses['style_loss_256'] = args.STYLE_FORE_ALPHA * loss_caculator.style_loss(x_neg_256, layers=layers)
        else:
            losses['style_loss_256'] = 0.
        if args.BACKGROUND_LOSS:
            layers = {'relu1_1': 0.2, 'relu2_1': 0.2, 'relu3_1': 0.2, 'relu4_1': 0.2, 'relu5_1': 0.2}
            losses['content_loss_256'] += args.CONTENT_BACK_ALPHA * loss_caculator.content_loss(out_256, layers=layers)
            # layers = {'pool1': 0.33, 'pool2': 0.34, 'pool3': 0.33}
            losses['style_loss_256'] += args.STYLE_BACK_ALPHA * loss_caculator.style_loss(out_256, layers=layers)

        if args.TV_ALPHA > 0:
            losses['tv_loss_256'] = loss_caculator.tv_loss(x_neg_256)
        else:
            losses['tv_loss_256'] = 0.

        # patch-gan-loss
        x_pos_neg_256 = tf.concat([x_pos_256, x_neg_256], axis=0)  # input as pos-neg to global discriminator
        pos_neg_256 = self.build_discriminator_256(x_pos_neg_256, name='discriminator256', sn=args.SN)
        pos_256, neg_256 = tf.split(pos_neg_256, 2)

        g_loss_256, d_loss_256, d_loss_real_256, d_loss_fake_256 = patch_gan_loss(pos_256, neg_256, name='patch_gan_loss256', loss_type=args.GAN_LOSS_TYPE)
        losses['g_loss_256'] = g_loss_256
        losses['d_loss_256'] = d_loss_256

        # gp loss (default not used)
        if args.GP_ALPHA > 0:
            interpolates = random_interpolates(x_pos_256, x_neg_256)    # interpolate
            dout = self.build_discriminator_256(interpolates, name='discriminator256', reuse=True, sn=args.SN)
            penalty = gradients_penalty(interpolates, dout, mask=mask_256)    # apply penalty
            gp_loss = penalty
        else:
            gp_loss = 0
        losses['gp_loss_256'] = gp_loss

        self.losses_256 = [losses['l1_loss_fore_256'],
                           losses['l1_loss_back_256'],
                           losses['grad_l1_loss_256'],
                           losses['edge_l1_loss_256'],
                           losses['grad_matching_256'],
                           losses['img_matching_256']
                           ]
        # Summary
        viz_img_256 = [x, x_incomplete_256, x_complete_256]
        viz_grad_256 = [grad_256[:, :, :, 0:1], grad_incomplete_256[:, :, :, 0:1], grad_complete_256[:, :, :, 0:1]]

        self.img_256 = tf.concat(viz_img_256, axis=2)
        self.grad_256 = tf.concat(viz_grad_256, axis=2)

        all_sum_256 = [tf.summary.scalar("l1_loss_fore_256", losses['l1_loss_fore_256']),
                       tf.summary.scalar("l1_loss_back_256", losses['l1_loss_back_256']),
                       tf.summary.scalar('g_loss_256', g_loss_256),
                       tf.summary.scalar('d_loss_256', d_loss_256),
                       tf.summary.scalar('d_loss_fake_256', d_loss_fake_256),
                       tf.summary.scalar('d_loss_real_256', d_loss_real_256),
                       tf.summary.scalar('content_loss_256', losses['content_loss_256']),
                       tf.summary.scalar('style_loss_256', losses['style_loss_256']),
                       tf.summary.image('raw_incomplete_predicted_complete_256',
                                        tf.concat(viz_img_256, axis=2), max_outputs=args.VIZ_MAX_OUT),
                       tf.summary.image('raw_incomplete_predicted_completed_grad_256',
                                        tf.concat(viz_grad_256, axis=2), max_outputs=args.VIZ_MAX_OUT),
                       tf.summary.scalar('tv_loss_256', losses['tv_loss_256']),
                       tf.summary.scalar('grad_l1_loss_256', losses['grad_l1_loss_256']),
                       tf.summary.scalar('edge_l1_loss_256', losses['edge_l1_loss_256']),
                       tf.summary.scalar('grad_matching_256', losses['grad_matching_256']),
                       tf.summary.scalar('img_matching_256', losses['img_matching_256']),
                       ]

        """##### Variables #####"""
        # generator vars
        self.total_g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        self.total_d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

        """##### Training Ops #####"""
        # train ops
        # Scale 64
        losses['total_g_loss_64'] = 0.0 *losses['l1_loss_fore_64'] + \
                                    0.0 *losses['l1_loss_back_64'] + \
                                    args.ALPHA * (losses['edge_l1_loss_64'] +
                                                   losses['grad_l1_loss_64'] +
                                                   losses['grad_matching_64'])

        # Scale 128
        losses['total_g_loss_128'] = 0.0*losses['l1_loss_fore_128'] + \
                                     0.0*losses['l1_loss_back_128'] + \
                                     args.ALPHA * (losses['edge_l1_loss_128'] +
                                                   losses['grad_l1_loss_128'] +
                                                   losses['grad_matching_128'])

        # Scale 256
        losses['total_g_loss_256'] = losses['l1_loss_fore_256'] + \
                                     losses['l1_loss_back_256'] + \
                                     losses['content_loss_256'] + \
                                     losses['style_loss_256'] + \
                                     args.PATCH_GAN_ALPHA * losses['g_loss_256'] + \
                                     args.TV_ALPHA * losses['tv_loss_256'] + \
                                     args.ALPHA * (losses['edge_l1_loss_256'] +
                                                   losses['grad_l1_loss_256'] +
                                                   losses['grad_matching_256']) + \
                                     losses['img_matching_256']

        losses['total_d_loss_256'] = losses['d_loss_256'] + args.GP_ALPHA * losses['gp_loss_256']

        self.g_loss = losses['total_g_loss_256'] + losses['total_g_loss_128'] + losses['total_g_loss_64']
        # self.g_loss = losses['total_g_loss_256']    # without deep structure supervision
        self.d_loss = losses['total_d_loss_256']

        # summary
        self.all_sum_64 = tf.summary.merge(all_sum_64)
        self.all_sum_128 = tf.summary.merge(all_sum_128)
        self.all_sum_256 = tf.summary.merge(all_sum_256)
        self.all_sum = tf.summary.merge(all_sum_64+all_sum_128+all_sum_256)


    def build_validation_model(self, x, mask, args, training=False, reuse=True):
        # Orgin image, edge
        image, edge, edge_128, edge_64 = x
        grad = tf.image.sobel_edges(image)  # normalization?
        grad = tf.reshape(grad, [args.VAL_NUM, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 6])  # 6 channel

        # x for image
        x = tf.reshape(image, [args.VAL_NUM, args.IMG_SHAPES[0], args.IMG_SHAPES[1],
                               args.IMG_SHAPES[2]])  # [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]]
        mask = tf.reshape(mask, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        edge = tf.reshape(edge, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        edge_128 = tf.reshape(edge_128, [-1, 128, 128, 1])
        edge_64 = tf.reshape(edge_64, [-1, 64, 64, 1])

        x_incomplete = x * (1. - mask)

        # incomplete edge at full scale
        input_edge = 1 - edge
        edge_incomplete = input_edge * (1 - mask) + mask  # 0 (black) for edge when save and input, 1 (white) for non edge

        # grad
        grad_incomplete = (1. - mask) * grad

        # bulid inpaint net
        out_256, out_64, out_128, out_grad_256, out_grad_64, out_grad_128 = self.build_inpaint_net(x_incomplete,
                                                                                                   edge_incomplete,
                                                                                                   grad_incomplete,
                                                                                                   mask, args,
                                                                                                   reuse=reuse,
                                                                                                   training=training,
                                                                                                   padding=args.PADDING)

        metrics = {}  # use a dict to collect metrics

        # TODO: scale 64
        # apply mask and complete image
        mask_64 = tf.image.resize_images(mask, (64, 64))
        x_incomplete_64 = tf.image.resize_images(x_incomplete, (64, 64))
        x_complete_64 = out_64 * mask_64 + x_incomplete_64

        x_64 = inverse_transform(tf.image.resize_images(x, (64, 64)))
        x_complete_64 = inverse_transform(x_complete_64)
        metrics['psnr_64'] = psnr(x_64, x_complete_64)
        metrics['ssmi_64'] = ssmi(x_64, x_complete_64)
        # self.metrics['mm-ssmi'] = mm_ssmi(x_128, x_complete_128)
        metrics['l1_64'] = avg_l1(x_64, x_complete_64)
        metrics['tv_64'] = tv_loss(x_complete_64)

        # summary
        viz_val_img_64 = [x_64, inverse_transform(x_incomplete_64), x_complete_64]
        self.val_img_64 = tf.concat(viz_val_img_64, axis=2)
        self.val_metrics_64 = [
            metrics['psnr_64'],
            metrics['ssmi_64'],
            metrics['l1_64']
        ]

        self.val_all_sum_64 = tf.summary.merge(
            [tf.summary.scalar('val_psnr_64', metrics['psnr_64']),
             tf.summary.scalar('val_ssmi_64', metrics['ssmi_64']),
             tf.summary.scalar('val_l1_64', metrics['l1_64']),
             tf.summary.scalar('val_tv_64', metrics['tv_64']),
             tf.summary.image('val_incomplete_predicted_complete_64', tf.concat(viz_val_img_64, axis=2),
                              max_outputs=args.VIZ_MAX_OUT)
             ]
        )

        # TODO: scale 128
        # apply mask and complete image
        mask_128 = tf.image.resize_images(mask, (128, 128))
        x_incomplete_128 = tf.image.resize_images(x_incomplete, (128, 128))
        x_complete_128 = out_128 * mask_128 + x_incomplete_128

        x_128 = inverse_transform(tf.image.resize_images(x, (128, 128)))
        x_complete_128 = inverse_transform(x_complete_128)
        metrics['psnr_128'] = psnr(x_128, x_complete_128)
        metrics['ssmi_128'] = ssmi(x_128, x_complete_128)
        # self.metrics['mm-ssmi'] = mm_ssmi(x_128, x_complete_128)
        metrics['l1_128'] = avg_l1(x_128, x_complete_128)
        metrics['tv_128'] = tv_loss(x_complete_128)

        # summary
        viz_val_img_128 = [x_128, inverse_transform(x_incomplete_128), x_complete_128]
        self.val_img_128 = tf.concat(viz_val_img_128, axis=2)
        self.val_metrics_128 = [
            metrics['psnr_128'],
            metrics['ssmi_128'],
            metrics['l1_128']
        ]

        self.val_all_sum_128 = tf.summary.merge(
            [tf.summary.scalar('val_psnr_128', metrics['psnr_128']),
             tf.summary.scalar('val_ssmi_128', metrics['ssmi_128']),
             tf.summary.scalar('val_l1_128', metrics['l1_128']),
             tf.summary.scalar('val_tv_128', metrics['tv_128']),
             tf.summary.image('val_incomplete_predicted_complete_128', tf.concat(viz_val_img_128, axis=2),
                              max_outputs=args.VIZ_MAX_OUT)
             ]
        )

        # TODO: scale 256
        # apply mask and complete image
        mask_256 = mask
        x_incomplete_256 = x_incomplete
        x_complete_256 = out_256 * mask_256 + x_incomplete_256

        x_256 = inverse_transform(x)
        x_complete_256 = inverse_transform(x_complete_256)
        metrics['psnr_256'] = psnr(x_256, x_complete_256)
        metrics['ssmi_256'] = ssmi(x_256, x_complete_256)
        # self.metrics['mm-ssmi'] = mm_ssmi(x_128, x_complete_128)
        metrics['l1_256'] = avg_l1(x_256, x_complete_256)
        metrics['tv_256'] = tv_loss(x_complete_256)

        # edge and grad loss
        grad_256 = grad
        grad_incomplete_256 = grad_incomplete
        grad_complete_256 = out_grad_256 * mask_256 + grad_incomplete_256

        # more weight for edges?
        # edge weight
        edge_mask_256 = edge  # 1 for edge, 0 for grad, when using feature.canny()
        mask_priority_256 = priority_loss_mask(edge_mask_256, ksize=5, sigma=1, iteration=2)
        edge_weight_256 = args.EDGE_ALPHA * mask_priority_256  # salient edge
        # grad weight
        grad_weight_256 = args.GRAD_ALPHA  # equaled grad

        # error
        grad_error_256 = tf.abs(out_grad_256 - grad_256)

        # edge loss
        metrics['edge_l1_loss_256'] = tf.reduce_sum(edge_weight_256 * grad_error_256) / tf.reduce_sum(edge_weight_256) / 6.

        # grad pixel level reconstruction loss
        metrics['grad_l1_loss_256'] = tf.reduce_mean(grad_weight_256 * grad_error_256)

        # grad patch level loss with implicit nearest neighbor matching
        metrics['grad_matching_256'] = grad_matching_loss(out_grad_256, grad_256, args)

        # summary
        # ones_x = tf.ones_like(x)
        # viz_val_img_256 = [x_256, inverse_transform(x_incomplete_256), x_complete_256,
        #                ones_x * edge_256, ones_x * edge_incomplete_256, ones_x * edge_complete_256]
        viz_val_grad_256 = [grad[:, :, :, 0:1], grad_incomplete_256[:, :, :, 0:1],
                            grad_complete_256[:, :, :, 0:1]]  # , out_grad_256[:,:,:,0:1]
        viz_val_img_256 = [x_256, inverse_transform(x_incomplete_256), x_complete_256]

        self.val_img_256 = tf.concat(viz_val_img_256, axis=2)
        self.val_metrics_256 = [
            metrics['psnr_256'],
            metrics['ssmi_256'],
            metrics['l1_256']
        ]

        self.val_all_sum_256 = tf.summary.merge(
            [tf.summary.scalar('val_psnr_256', metrics['psnr_256']),
             tf.summary.scalar('val_ssmi_256', metrics['ssmi_256']),
             tf.summary.scalar('val_l1_256', metrics['l1_256']),
             tf.summary.scalar('val_grad_l1_loss_256', metrics['grad_l1_loss_256']),
             tf.summary.scalar('val_grad_matching_256', metrics['grad_matching_256']),
             tf.summary.scalar('val_edge_l1_loss_256', metrics['edge_l1_loss_256']),
             tf.summary.image('val_incomplete_predicted_complete_256', tf.concat(viz_val_img_256, axis=2),
                              max_outputs=12),
             tf.summary.image('val_incomplete_predicted_completed_grad_256',
                              tf.concat(viz_val_grad_256, axis=2), max_outputs=args.VIZ_MAX_OUT),
             ]
        )

    def build_test_model(self, x, mask, args, training=False, reuse=False):
        # Orgin image, edge
        image, edge  = x
        grad = tf.image.sobel_edges(image)  # normalization?
        grad = tf.reshape(grad, [args.TEST_NUM, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 6])  # 6 channel

        # x for image
        x = tf.reshape(image, [args.TEST_NUM, args.IMG_SHAPES[0], args.IMG_SHAPES[1],
                               args.IMG_SHAPES[2]])  # [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]]
        mask = tf.reshape(mask, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        edge = tf.reshape(edge, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])

        # incomplete image
        x_incomplete = x * (1. - mask)

        # incomplete edge at full scale
        input_edge = 1 - edge
        print(edge.shape)
        print(mask.shape)
        edge_incomplete = input_edge * (1 - mask) + mask  # 0 (black) for edge when save and input, 1 (white) for non edge

        # grad
        grad_incomplete = (1. - mask) * grad

        # bulid inpaint net
        out_256, out_64, out_128, out_grad_256, out_grad_64, out_grad_128 = self.build_inpaint_net(x_incomplete,
                                                                                                   edge_incomplete,
                                                                                                   grad_incomplete,
                                                                                                   mask, args,
                                                                                                   reuse=reuse,
                                                                                                   training=training,
                                                                                                   padding=args.PADDING)

        # scale 256
        self.raw_x = inverse_transform(x)
        self.raw_x_incomplete = self.raw_x*(1-mask)
        self.raw_x_complete = self.raw_x_incomplete + inverse_transform(out_256)*mask
        self.mask = mask

        self.psnr = tf.image.psnr(self.raw_x, self.raw_x_complete, max_val=255)
        self.ssim = tf.image.ssim(self.raw_x, self.raw_x_complete, max_val=255)
        self.l1 = tf.reduce_mean(tf.abs(self.raw_x - self.raw_x_complete), axis=[1, 2, 3])

        # scale 128
        self.raw_x_128 = tf.image.resize_nearest_neighbor(self.raw_x, (128,128))
        self.mask_128 = tf.image.resize_nearest_neighbor(self.mask, (128,128))
        self.raw_x_incomplete_128 = self.raw_x_128 * (1 - self.mask_128)
        self.raw_x_complete_128 = self.raw_x_incomplete_128 + inverse_transform(out_128) * self.mask_128

        self.psnr_128 = tf.image.psnr(self.raw_x_128, self.raw_x_complete_128, max_val=255)
        self.ssim_128 = tf.image.ssim(self.raw_x_128, self.raw_x_complete_128, max_val=255)
        self.l1_128 = tf.reduce_mean(tf.abs(self.raw_x_128 - self.raw_x_complete_128), axis=[1, 2, 3])

        # scale 64
        self.raw_x_64 = tf.image.resize_nearest_neighbor(self.raw_x, (64,64))
        self.mask_64 = tf.image.resize_nearest_neighbor(self.mask, (64,64))
        self.raw_x_incomplete_64 = self.raw_x_64 * (1 - self.mask_64)
        self.raw_x_complete_64 = self.raw_x_incomplete_64 + inverse_transform(out_64) * self.mask_64

        self.psnr_64 = tf.image.psnr(self.raw_x_64, self.raw_x_complete_64, max_val=255)
        self.ssim_64 = tf.image.ssim(self.raw_x_64, self.raw_x_complete_64, max_val=255)
        self.l1_64 = tf.reduce_mean(tf.abs(self.raw_x_64 - self.raw_x_complete_64), axis=[1, 2, 3])
        # return raw_x, raw_x_incomplete, raw_x_complete, mask

    def evaluate(self, x, mask, args, training=False, reuse=False):
        # Orgin image, edge
        image, edge  = x
        grad = tf.image.sobel_edges(image)  # normalization?
        grad = tf.reshape(grad, [1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 6])  # 6 channel

        # x for image
        x = tf.reshape(image, [1, args.IMG_SHAPES[0], args.IMG_SHAPES[1],
                               args.IMG_SHAPES[2]])  # [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]]
        mask = tf.reshape(mask, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        edge = tf.reshape(edge, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])

        # incomplete image
        x_incomplete = x * (1. - mask)

        # incomplete edge at full scale
        input_edge = 1 - edge
        print(edge.shape)
        print(mask.shape)
        edge_incomplete = input_edge * (1 - mask) + mask  # 0 (black) for edge when save and input, 1 (white) for non edge

        # grad
        grad_incomplete = (1. - mask) * grad

        # bulid inpaint net
        out_256, out_64, out_128, out_grad_256, out_grad_64, out_grad_128 = self.build_inpaint_net(x_incomplete,
                                                                                                   edge_incomplete,
                                                                                                   grad_incomplete,
                                                                                                   mask, args,
                                                                                                   reuse=reuse,
                                                                                                   training=training,
                                                                                                   padding=args.PADDING)