import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
#from utils_fn import *
from ops import *
import time

class InpaintModel():

    def __init__(self, args):
        self.model_name = "InpaintModel"   # name for checkpoint
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

        return x


    def evaluate(self, x, edge,  mask, args, training=False, reuse=False):
        # image, grad map
        image = normalize(x)
        grad = tf.image.sobel_edges(image)  # normalization?
        grad = tf.reshape(grad, [1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 6])  # 6 channel

        # x for image
        x = tf.reshape(image, [1, args.IMG_SHAPES[0], args.IMG_SHAPES[1],
                               args.IMG_SHAPES[2]])  # [1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], args.IMG_SHAPES[2]]
        mask = tf.reshape(mask, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])
        edge = tf.reshape(edge, [-1, args.IMG_SHAPES[0], args.IMG_SHAPES[1], 1])

        # incomplete image
        x_incomplete = x * (1. - mask)

        # incomplete edge at full scale
        input_edge = 1 - edge
        edge_incomplete = input_edge * (1 - mask) + mask  # 0 (black) for edge when save and input, 1 (white) for non edge

        # grad
        grad_incomplete = (1. - mask) * grad

        out_256 = self.build_inpaint_net(x_incomplete, edge_incomplete, grad_incomplete, 
                                          mask, args, reuse=reuse,training=training, padding=args.PADDING)

        raw_x = inverse_transform(x)
        raw_x_incomplete = raw_x * (1 - mask)
        raw_x_complete = raw_x_incomplete + inverse_transform(out_256) * mask

        return raw_x_complete
