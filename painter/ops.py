import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv', IN=True, reuse=False,
             padding='SAME', activation=tf.nn.elu, use_bias=True, training=True, sn=False):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFLECT']
    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        """ 
        Padding layer.
        Dilated kernel size: k_r = ksize + (rate - 1)*(ksize - 1)
        Padding size: o = i + 2p - k_r and o = i, so p = rate * (ksize - 1) / 2 (when i and o has the same image shape)
        """
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'

    # if spectrum normalization
    if sn:
        with tf.variable_scope(name, reuse=reuse):
            w = tf.get_variable("kernel", shape=[ksize, ksize, x.get_shape()[-1], cnum], initializer=weight_init,
                                regularizer=weight_regularizer)

            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding=padding, dilations=[1, rate, rate, 1])
            if use_bias:
                bias = tf.get_variable("bias", [cnum], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
    else:
        x = tf.layers.conv2d(inputs=x, filters=cnum, activation=None,
                             kernel_size=ksize, strides=stride,
                             dilation_rate=rate, padding=padding,
                             kernel_initializer=None,
                             kernel_regularizer=weight_regularizer,
                             use_bias=use_bias)
    if IN:
        x = tf.contrib.layers.instance_norm(x)    # if instance norm? before non-linear activation!!!
    if activation is not None:
        x = activation(x)
    return x

@add_arg_scope
def gen_deconv(x, cnum, ksize=4, stride=2, rate=1, method='deconv',IN=True,
               activation=tf.nn.relu, name='upsample', padding='SAME', sn=False, training=True, reuse=False):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name, reuse=reuse):
        if method == 'nearest':
            x = resize(x, func=tf.image.resize_nearest_neighbor)    # tf.image.resize_bilinear ?
            x = gen_conv(
                x, cnum, 3, 1, name=name+'_conv', padding=padding,
                training=training, IN=IN)
        elif method == 'bilinear':
            x = resize(x, func=tf.image.resize_bilinear)
            x = gen_conv(
                x, cnum, 3, 1, name=name + '_conv', padding=padding,
                training=training, IN=IN)
        elif method == 'bicubic':
            x = resize(x, func=tf.image.resize_bicubic)
            x = gen_conv(
                x, cnum, 3, 1, name=name + '_conv', padding=padding,
                training=training, IN=IN)    # default instance normalization, see function gen_conv()
        else:
            # assert padding in ['SYMMETRIC', 'SAME', 'REFLECT']
            # if padding == 'SYMMETRIC' or padding == 'REFLECT':
            #     p = int(rate * (ksize - 1) / 2)
            #     p = 0
            #     x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
            padding = 'SAME'
            x = tf.layers.conv2d_transpose(x, cnum, kernel_size=ksize, strides=stride,
                                           activation=None, padding=padding)
            if IN:
                x = tf.contrib.layers.instance_norm(x)    # if instance norm?
            if activation is not None:
                x = activation(x)
    return x

def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x

# yj
@add_arg_scope
def resnet_blocks(x, cnum, ksize, stride, rate, block_num, name, IN=True,
                  padding='REFLECT', activation=tf.nn.elu, training=True):
    for block in range(block_num):
        # x = resnet_block12(x, cnum, ksize, stride, rate, name+"_"+str(block), padding, activation, training=training)
        x = resnet_block21(x, cnum, ksize, stride, rate, name + "_" + str(block), padding=padding,
                           activation=activation, training=training)
    return x

# yj
def resnet_block21(x, cnum, ksize, stride, rate, name, IN=True,
                  padding='SAME', activation=tf.nn.relu, training=True):
    xin = x
    assert padding in ['SYMMETRIC', 'SAME', 'REFLECT']
    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding1 = 'VALID'
    else:
        padding1 = padding
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding1, name=name+"0")
    if IN:
        x = tf.contrib.layers.instance_norm(x)    # if instance norm?
    if activation is not None:
        x = activation(x)

    rate = 1
    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding2 = 'VALID'
    else:
        padding2 = padding
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding2, name=name+"1")
    if IN:
        x = tf.contrib.layers.instance_norm(x)    # if instance norm?
    return xin + x

# yj
def resnet_block12(x, cnum, ksize, stride, rate, name, IN=True,
                  padding='REFLECT', activation=tf.nn.elu, training=True):
    xin = x
    rate = 1
    assert padding in ['SYMMETRIC', 'SAME', 'REFLECT']
    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding1 = 'VALID'
    else:
        padding1 = padding
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding1, name=name+"0")
    if IN:
        x = tf.contrib.layers.instance_norm(x)    # if instance norm?
    if activation is not None:
        x = activation(x)

    rate = 2
    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding2 = 'VALID'
    else:
        padding2 = padding
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding2, name=name+"1")
    if IN:
        x = tf.contrib.layers.instance_norm(x)    # if instance norm?

    return xin + x


# TODO：torgb, only with conv 1x1 and bias are enough? 线性输出 vs 使用tanh激活函数
def torgb(x, cnum, ksize, stride, rate, name, activation=tf.nn.tanh, padding="SAME"):
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    # x = tf.clip_by_value(x, -1., 1.)
    return x


def dis_conv(x, cnum, ksize=5, stride=2, rate=1, activation=tf.nn.leaky_relu, name='conv',
             padding='SAME', use_bias=True, sn=True, training=True, reuse=False):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    # if spectrum normalization
    if sn:
        with tf.variable_scope(name, reuse=reuse):
            w = tf.get_variable("kernel", shape=[ksize, ksize, x.get_shape()[-1], cnum], initializer=weight_init,
                                regularizer=weight_regularizer)

            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding=padding, dilations=[1, rate, rate, 1])
            if use_bias:
                bias = tf.get_variable("bias", [cnum], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
            if activation is not None:
                x = activation(x)
    else:
        x = tf.layers.conv2d(inputs=x, filters=cnum, activation=activation,
                             kernel_size=ksize, strides=stride,
                             dilation_rate=rate, padding=padding,
                             kernel_initializer=None,
                             kernel_regularizer=None,
                             use_bias=use_bias,
                             reuse=reuse)
    return x

def flatten(x, name='flatten'):
    """Flatten wrapper.
    """
    with tf.variable_scope(name):
        return tf.contrib.layers.flatten(x)

def out_complete(out, x_incomplete, mask, res):
    mask = tf.image.resize_images(mask, (res, res))
    x_incomplete = tf.image.resize_images(x_incomplete, (res, res))
    x_complete = out * mask + x_incomplete * (1. - mask)
    return x_complete


# linear embedding
@add_arg_scope
def conv(x, channels, kernel=3, stride=1, pad=0, pad_type='REFLECT', use_bias=True, sn=False, scope='conv_0', reuse=False, training=False, padding=None):
    with tf.variable_scope(scope, reuse=reuse):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias, reuse=reuse)
        return x

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def max_pooling(x, pool_size=2):
    x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=pool_size, padding='SAME')
    return x


def avg_pooling(x, pool_size=2):
    x = tf.layers.average_pooling2d(x, pool_size=pool_size, strides=pool_size, padding='SAME')
    return x

# ATN layer
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

@add_arg_scope
def AtnConv(x1, x2, mask=None, ksize=3, stride=1, rate=2,
            softmax_scale=10., training=True, rescale=False):
    r""" Attention transfer networks implementation in tensorflow

    Attention transfer networks is introduced in publication:
      Learning Pyramid-Context Encoder Networks for High-Quality Image Inpainting, Zeng et al.
      https://arxiv.org/pdf/1904.07475.pdf
      https://github.com/researchmm/PEN-Net-for-Inpainting
    inspired by:
      Generative Image Inpainting with Contextual Attention, Yu et al.
      https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
      https://arxiv.org/abs/1801.07892
    Args:
      x1:  low-level feature map with larger  size [b, h, w, c].
      x2: high-level feature map with smaller size [b, h/2, w/2, c].
      mask: Input mask, 1 for missing regions 0 for known regions.
      ksize: Kernel size for attention transfer networks.
      stride: Stride for extracting patches from feature map.
      rate: Dilation for matching.
      softmax_scale: Scaled softmax for attention.
      training: Indicating if current graph is training or inference.
      rescale: Indicating if input feature maps need to be downsample
    Returns:
      tf.Tensor: reconstructed feature map
    """
    # downsample input feature maps if needed due to limited GPU memory
    if rescale:
        x1 = resize(x1, scale=1. / 2, func=tf.image.resize_nearest_neighbor)
        x2 = resize(x2, scale=1. / 2, func=tf.image.resize_nearest_neighbor)
    # get shapes
    raw_x1s = tf.shape(x1)
    int_x1s = x1.get_shape().as_list()
    int_x2s = x2.get_shape().as_list()

    # extract patches from low-level feature maps for reconstruction
    kernel = 2 * rate
    raw_w = tf.extract_image_patches(
        x1, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')
    raw_w = tf.reshape(raw_w, [int_x1s[0], -1, kernel, kernel, int_x1s[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to [b, kernel, kernel, c, hw]
    raw_w_groups = tf.split(raw_w, int_x1s[0], axis=0)

    # extract patches from high-level feature maps for matching and attending
    x2_groups = tf.split(x2, int_x2s[0], axis=0)
    w = tf.extract_image_patches(
        x2, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    w = tf.reshape(w, [int_x2s[0], -1, ksize, ksize, int_x2s[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to [b, ksize, ksize, c, hw/4]    # need transpose?? -- 480
    w_groups = tf.split(w, int_x2s[0], axis=0)

    # resize and extract patches from masks
    mask = resize(mask, to_shape=int_x2s[1:3], func=tf.image.resize_nearest_neighbor)
    m = tf.extract_image_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to [1, ksize, ksize, 1, hw/4]
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0, 1, 2], keep_dims=True), 0.), tf.float32)

    # matching and attending hole and non-hole patches
    y = []
    scale = softmax_scale
    # high level patches: w_groups, low level patches: raw_w_groups, x2_groups: high level feature map
    for xi, wi, raw_wi in zip(x2_groups, w_groups, raw_w_groups):
        # matching on high-level feature maps
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0, 1, 2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")
        yi = tf.reshape(yi, [1, int_x2s[1], int_x2s[2], (int_x2s[1] // stride) * (int_x2s[2] // stride)])
        # apply softmax to obtain attention score
        yi *= mm  # mask
        yi = tf.nn.softmax(yi * scale, 3)
        yi *= mm  # mask    yi: score maps, score maps for non-hole regions are zeros through masks
        # transfer non-hole features into holes according to the atttention score
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_x1s[1:]], axis=0),
                                    strides=[1, rate * stride, rate * stride, 1]) / 4.    # filter: [height, width, output_channels, in_channels]
        y.append(yi)
    y = tf.concat(y, axis=0)
    y.set_shape(int_x1s)
    # refine filled feature map after matching and attending
    y1 = tf.layers.conv2d(y, int_x1s[-1] // 4, 3, 1, dilation_rate=1, activation=tf.nn.relu, padding='SAME')
    y2 = tf.layers.conv2d(y, int_x1s[-1] // 4, 3, 1, dilation_rate=2, activation=tf.nn.relu, padding='SAME')
    y3 = tf.layers.conv2d(y, int_x1s[-1] // 4, 3, 1, dilation_rate=4, activation=tf.nn.relu, padding='SAME')
    y4 = tf.layers.conv2d(y, int_x1s[-1] // 4, 3, 1, dilation_rate=8, activation=tf.nn.relu, padding='SAME')
    y = tf.concat([y1, y2, y3, y4], axis=3)
    if rescale:
        y = resize(y, scale=2., func=tf.image.resize_nearest_neighbor)
    return y


"""##### our-attention #####"""
def attention(x, channels, neighbors=1, use_bias=True, sn=False, down_scale = 2, pool_scale=2, 
              name='attention_pooling', training=True, padding='REFLECT', reuse=False):
    if neighbors > 1:
        x = attention_with_neighbors(x, channels, down_scale=down_scale, pool_scale=pool_scale, name=name)
    else:
        x = attention_with_pooling(x, channels, down_scale=down_scale, pool_scale=pool_scale, name=name)
    return x
    
@add_arg_scope
def attention_with_pooling(x, channels, ksize=4, use_bias=True, sn=False, down_scale = 2, pool_scale=2, name='attention_pooling', training=True, padding='REFLECT', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x_origin = x

        # down sampling
        if down_scale > 1:
            x = gen_conv(x, channels, ksize, stride=down_scale, activation=tf.nn.relu, name='attention_down_sample',reuse=reuse)

        # attention
        f = conv(x, channels // 16, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='f_conv', reuse=reuse)  # [bs, h, w, c']
        f = max_pooling(f, pool_scale)
        # f = avg_pooling(f)

        g = conv(x, channels // 16, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='g_conv',reuse=reuse)  # [bs, h, w, c']

        h = conv(x, channels // 16, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='h_conv',reuse=reuse)  # [bs, h, w, c]
        h = max_pooling(h, pool_scale)
        # h = avg_pooling(h)   [4,65536,4096]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 16])  # [bs, h, w, C]
        # o = conv(o, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='attn_conv_up')    # from bottleneck

        # up sampling
        if down_scale > 1:
            o = gen_deconv(o, channels, ksize, method='deconv', stride=down_scale, activation=tf.nn.relu, name='attention_down_upsample',reuse=reuse)

        x = gamma * o + x_origin

    return x

# attention consider neighbors
@add_arg_scope
def attention_with_neighbors(x, channels, ksize=3, use_bias=True, sn=False, stride=2,
                           down_scale = 2, pool_scale=2, name='attention_pooling',
                           training=True, padding='REFLECT', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x1 = x

        # downsample input feature maps if needed due to limited GPU memory
        # down sampling
        if down_scale > 1:
            x1 = gen_conv(x1, channels, ksize, stride=down_scale, activation=tf.nn.relu, name='attention_down_sample',
                         reuse=reuse)
        # get shapes
        int_x1s = x1.get_shape().as_list()
         # extract patches from high-level feature maps for matching and attending
        x1_groups = tf.split(x1, int_x1s[0], axis=0)
        w = tf.extract_image_patches(
            x1, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
        w = tf.reshape(w, [int_x1s[0], -1, ksize, ksize, int_x1s[3]])
        w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to [b, ksize, ksize, c, hw/4]    # need transpose?? -- 480
        w_groups = tf.split(w, int_x1s[0], axis=0)

        # matching and attending hole and non-hole patches
        y = []
        scale = 10.
        # high level patches: w_groups, low level patches: raw_w_groups, x2_groups: high level feature map
        for xi, wi in zip(x1_groups, w_groups):
            # matching on high-level feature maps
            wi = wi[0]
            wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0, 1, 2])), 1e-4)
            yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")
            yi = tf.reshape(yi, [1, int_x1s[1], int_x1s[2], (int_x1s[1] // stride) * (int_x1s[2] // stride)])
            yi = tf.nn.softmax(yi * scale, 3)
            # non local mean
            wi_center = tf.transpose(wi, [0, 1, 3, 2])
            yi = tf.nn.conv2d(yi, wi_center, strides=[1, 1, 1, 1], padding="SAME") / 4.

            # filter: [height, width, output_channels, in_channels]
            y.append(yi)
        y = tf.concat(y, axis=0)
        y.set_shape(int_x1s)
        # up sampling
        if down_scale > 1:
            y = gen_deconv(y, channels, ksize, method='deconv', stride=down_scale, activation=tf.nn.relu,
                           name='attention_down_upsample', reuse=reuse)

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        x = gamma * y + x
        x = tf.layers.conv2d(x, channels, 3, 1, dilation_rate=1, activation=tf.nn.relu, padding='SAME')
    return x
    
def normalize(x) :
    return x/127.5 - 1
    
def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)*127.5