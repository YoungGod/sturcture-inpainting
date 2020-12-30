import vgg_network
from logging import exception
import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict

from sys import stdout
from functools import reduce
from vgg_network import VGG


# loss config
config = edict()
config.W = edict()

# TODO: content
# weights
config.W.Content = 1.

config.Content = edict()
config.Content.feat_layers = {'relu1_1': 0.2, 'relu2_1': 0.2,'relu3_1': 0.2,'relu4_1': 0.2,'relu5_1': 0.2}

# TODO: style
config.W.Style = 1.
config.Style = edict()
config.Style.feat_layers = {'relu1_1': 0.2, 'relu2_1': 0.2,'relu3_1': 0.2,'relu4_1': 0.2,'relu5_1': 0.2}


class LossCalculator:

    def __init__(self, vgg_dir, real_image):
        self.vgg_model = VGG(vgg_dir)
        self.vgg_real = self.vgg_model.net(real_image)

    def content_loss(self, content_fake, layers=None):
        # compute content loss
        vgg_fake = self.vgg_model.net(content_fake)    # dict: net[name] = current_layer
        if config.W.Content > 0:
            if layers is not None:
                config.Content.feat_layers = layers
            content_loss_list = [w * self._content_loss_helper(self.vgg_real[layer], vgg_fake[layer])
                            for layer, w in config.Content.feat_layers.items()]
            content_loss = tf.reduce_sum(content_loss_list)
        else:
            zero_tensor = tf.constant(0.0, dtype=tf.float32)
            content_loss = zero_tensor
        return content_loss

    def style_loss(self, style_fake, layers=None):
        vgg_fake = self.vgg_model.net(style_fake)  # dict: net[name] = current_layer
        # image = tf.placeholder('float32', shape=style.shape)
        # style_net = self.vgg.net(image)

        if config.W.Style > 0:
            if layers is not None:
                config.Style.feat_layers = layers
            style_loss_list = [w * self._style_loss_helper(self.vgg_real[layer], vgg_fake[layer])
                            for layer, w in config.Style.feat_layers.items()]
            style_loss = tf.reduce_sum(style_loss_list)
        else:
            zero_tensor = tf.constant(0.0, dtype=tf.float32)
            style_loss = zero_tensor
        return style_loss

    # def _calculate_input_gram_matrix_for(self, layer):
    #     image_feature = self.network[layer]
    #     _, height, width, number = map(lambda i: i.value, image_feature.get_shape())
    #     size = height * width * number
    #     image_feature = tf.reshape(image_feature, (-1, number))
    #     return tf.matmul(tf.transpose(image_feature), image_feature) / size


    def _content_loss_helper(self, vgg_A, vgg_B):
        N, fH, fW, fC = vgg_A.shape.as_list()
        feature_size = N * fH * fW *fC
        content_loss = 2 * tf.nn.l2_loss(vgg_A - vgg_B) / feature_size
        return content_loss

    def _style_loss_helper(self, vgg_A, vgg_B):
        N, fH, fW, fC = vgg_A.shape.as_list()
        feature_size = N * fH * fW *fC
        gram_A = self._compute_gram(vgg_A)
        gram_B = self._compute_gram(vgg_B)
        style_loss = 2 * tf.nn.l2_loss(gram_A - gram_B) / feature_size
        return style_loss

    def _compute_gram(self, feature):
        # https://github.com/fullfanta/real_time_style_transfer/blob/master/train.py
        shape = tf.shape(feature)
        psi = tf.reshape(feature, [shape[0], shape[1] * shape[2], shape[3]])
        # psi_t = tf.transpose(psi, perm=[0, 2, 1])
        gram = tf.matmul(psi, psi, transpose_a=True)
        gram = tf.div(gram, tf.cast(shape[1] * shape[2] * shape[3], tf.float32))
        return gram

    def tv_loss(self, image):
        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        shape = image.shape.as_list()
        tv_loss = 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))

        return tv_loss

    # TODO: l1_loss(x, x_complete_256)
    def l1_loss(self, image, predict, mask, type='foreground'):
        error = tf.abs(predict - image)
        if type == 'foreground':
            loss = tf.reduce_sum(mask * error) / tf.reduce_sum(mask)    # * tf.reduce_sum(1. - mask) for balance?
        elif type == 'background':
            loss = tf.reduce_sum((1. - mask) * error) / tf.reduce_sum(1. - mask)
        else:
            loss = tf.reduce_sum(mask * tf.abs(predict - image)) / tf.reduce_sum(mask)
        return loss

    # TODO:
    def adversarial_loss(self):
        pass

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def gan_wgan_loss(pos, neg, name='gan_wgan_loss'):
    """
    wgan loss function for GANs.

    - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    """
    with tf.variable_scope(name):
        d_loss = tf.reduce_mean(neg-pos)
        g_loss = -tf.reduce_mean(neg)
        # scalar_summary('d_loss', d_loss)
        # scalar_summary('g_loss', g_loss)
        # scalar_summary('pos_value_avg', tf.reduce_mean(pos))
        # scalar_summary('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss

def patch_gan_loss(pos, neg, name='patch_gan_loss', loss_type='gan'):
    """
    patch gan loss
    """
    with tf.variable_scope(name):
        if loss_type =='gan':
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg)))  # 生成器loss

            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.zeros_like(neg)))
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=pos, labels=tf.ones_like(pos)))
            d_loss = d_loss_fake + d_loss_real  # 判别器loss

        if loss_type == 'hinge':
            d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - pos))
            d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + neg))
            d_loss = d_loss_real + d_loss_fake

            g_loss = -tf.reduce_mean(neg)

    return g_loss, d_loss, d_loss_real, d_loss_fake

def random_interpolates(x, y, alpha=None):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def gradients_penalty(x, y, mask=None, norm=1.):
    """Improved Training of Wasserstein GANs

    - https://arxiv.org/abs/1704.00028
    """
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))

from tensorflow.python.ops import array_ops
def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor, z = 1.
        ref: https://github.com/ailias/Focal-Loss-implement-on-Tensorflow
        if z == 1, J = -a * (1 – p) * log(p)
        if z != 1, J = -(1 – a) * p * log(1 –p)
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent)

def sigmoid_cross_entropy_balanced_fore(logits, label, mask, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(mask * (1. - y))
    count_pos = tf.reduce_sum(mask * y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    # cost = tf.reduce_mean(cost * (1 - beta))
    # N, H, W, C = logits.get_shape().as_list()
    size = count_neg + count_neg
    cost = tf.reduce_sum(cost * (1 - beta)) / size

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)

def sigmoid_cross_entropy_balanced_back(logits, label, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)


"""
id-mrf
"""
from enum import Enum

class Distance(Enum):
    L2 = 0
    DotProduct = 1

class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=3):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = tf.exp((self.b - scaled_distances) / self.sigma, name='weights_before_normalization')
        self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)

    def reversed_direction_CS(self):
        cs_flow_opposite = CSFlow(self.sigma, self.b)
        cs_flow_opposite.raw_distances = self.raw_distances
        work_axis = [1, 2]
        relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
        cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
        return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.1), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        with tf.name_scope('CS'):
            sT = T_features.shape.as_list()
            sI = I_features.shape.as_list()

            Ivecs = tf.reshape(I_features, (sI[0], -1, sI[3]))
            Tvecs = tf.reshape(T_features, (sI[0], -1, sT[3]))
            r_Ts = tf.reduce_sum(Tvecs * Tvecs, 2)
            r_Is = tf.reduce_sum(Ivecs * Ivecs, 2)
            raw_distances_list = []

            N, _, _, _ = T_features.shape.as_list()
            for i in range(N):
                Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
                A = tf.matmul(Tvec,tf.transpose(Ivec))
                cs_flow.A = A
                # A = tf.matmul(Tvec, tf.transpose(Ivec))
                r_T = tf.reshape(r_T, [-1, 1])  # turn to column vector
                dist = r_T - 2 * A + r_I
                cs_shape = sI[:3] + [dist.shape[0].value]
                cs_shape[0] = 1
                dist = tf.reshape(tf.transpose(dist), cs_shape)
                # protecting against numerical problems, dist should be positive
                dist = tf.maximum(float(0.0), dist)
                # dist = tf.sqrt(dist)
                raw_distances_list += [dist]

            cs_flow.raw_distances = tf.convert_to_tensor([tf.squeeze(raw_dist, axis=0) for raw_dist in raw_distances_list])

            relative_dist = cs_flow.calc_relative_distances()
            cs_flow.__calculate_CS(relative_dist)
            return cs_flow

    #--
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(1.0), b=float(1.0), args=None):
        cs_flow = CSFlow(sigma, b)
        with tf.name_scope('CS'):
            # prepare feature before calculating cosine distance
            T_features, I_features = cs_flow.center_by_T(T_features, I_features)
            with tf.name_scope('TFeatures'):
                T_features = CSFlow.l2_normalize_channelwise(T_features)
            with tf.name_scope('IFeatures'):
                I_features = CSFlow.l2_normalize_channelwise(I_features)
                # work seperatly for each example in dim 1
                cosine_dist_l = []
                N, _, _, _ = T_features.shape.as_list()
                for i in range(N):
                    T_features_i = tf.expand_dims(T_features[i, :, :, :], 0)
                    I_features_i = tf.expand_dims(I_features[i, :, :, :], 0)
                    patches_i = cs_flow.patch_decomposition(T_features_i, args)
                    # every patch in patches_i as a kernel to conv I_features, obtain dis between each patch in patches_i
                    # and I_features. (GPU is OK?)
                    cosine_dist_i = tf.nn.conv2d(I_features_i, patches_i, strides=[1, 1, 1, 1],
                                                        padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
                    cosine_dist_l.append(cosine_dist_i)

                cs_flow.cosine_dist = tf.concat(cosine_dist_l, axis = 0)

                cosine_dist_zero_to_one = -(cs_flow.cosine_dist - 1) / 2
                cs_flow.raw_distances = cosine_dist_zero_to_one

                relative_dist = cs_flow.calc_relative_distances()
                cs_flow.__calculate_CS(relative_dist)
                return cs_flow

    def calc_relative_distances(self, axis=3):
        epsilon = 1e-5
        div = tf.reduce_min(self.raw_distances, axis=axis, keep_dims=True)
        # div = tf.reduce_mean(self.raw_distances, axis=axis, keep_dims=True)
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    def weighted_average_dist(self, axis=3):
        if not hasattr(self, 'raw_distances'):
            raise exception('raw_distances property does not exists. cant calculate weighted average l2')

        multiply = self.raw_distances * self.cs_NHWC
        return tf.reduce_sum(multiply, axis=axis, name='weightedDistPerPatch')

    # --
    @staticmethod
    def create(I_features, T_features, distance : Distance, nnsigma=float(1.0), b=float(1.0), args=None):
        if distance.value == Distance.DotProduct.value:
            cs_flow = CSFlow.create_using_dotP(I_features, T_features, nnsigma, b, args)
        elif distance.value == Distance.L2.value:
            cs_flow = CSFlow.create_using_L2(I_features, T_features, nnsigma, b)
        else:
            raise "not supported distance " + distance.__str__()
        return cs_flow

    @staticmethod
    def sum_normalize(cs, axis=3):
        reduce_sum = tf.reduce_sum(cs, axis, keep_dims=True, name='sum')
        return tf.divide(cs, reduce_sum, name='sumNormalized')

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size

        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT, self.varT = tf.nn.moments(
            T_features, axes, name='TFeatures/moments')
        # we do not divide by std since its causing the histogram
        # for the final cs to be very thin, so the NN weights
        # are not distinctive, giving similar values for all patches.
        # stdT = tf.sqrt(varT, "stdT")
        # correct places with std zero
        # stdT[tf.less(stdT, tf.constant(0.001))] = tf.constant(1)
        with tf.name_scope('TFeatures/centering'):
            self.T_features_centered = T_features - self.meanT
        with tf.name_scope('IFeatures/centering'):
            self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = tf.norm(features, ord='euclidean', axis=3, name='norm')
        # expanding the norms tensor to support broadcast division
        norms_expanded = tf.expand_dims(norms, 3)
        features = tf.divide(features, norms_expanded, name='normalized')
        return features

    def patch_decomposition(self, T_features, args=None):
        # patch decomposition
        if args is None:
            patch_size = 1
            stride_size = 1
        else:
            patch_size = args.PATCH_SIZE
            stride_size = args.STRIDE_SIZE
        patches_as_depth_vectors = tf.extract_image_patches(
            images=T_features, ksizes=[1, patch_size, patch_size, 1],
            strides=[1, stride_size, stride_size, 1], rates=[1, 1, 1, 1], padding='VALID',
            name='patches_as_depth_vectors')

        out_channels = int(patches_as_depth_vectors.shape[3].value / patch_size / patch_size)
        self.patches_NHWC = tf.reshape(
            patches_as_depth_vectors,
            shape=[-1, patch_size, patch_size, out_channels],
            name='patches_PHWC')    # patches_as_depth_vectors.shape[3].value / patch_size / patch_size; because here path_size=1,so it's right

        self.patches_HWCN = tf.transpose(
            self.patches_NHWC,
            perm=[1, 2, 3, 0],
            name='patches_HWCP')  # tf.conv2 ready format (every patch as a kernel)

        return self.patches_HWCN


def mrf_loss(T_features, I_features, distance=Distance.DotProduct, nnsigma=float(1.0), args=None):
    T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
    I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)

    with tf.name_scope('cx'):
        cs_flow = CSFlow.create(I_features, T_features, distance, nnsigma)
        # sum_normalize:
        height_width_axis = [1, 2]
        # To:
        cs = cs_flow.cs_NHWC
        k_max_NC = tf.reduce_max(cs, axis=height_width_axis)
        CS = tf.reduce_mean(k_max_NC, axis=[1])
        CS_as_loss = 1 - CS
        CS_loss = -tf.log(1 - CS_as_loss)
        CS_loss = tf.reduce_mean(CS_loss)
        return CS_loss


def random_sampling(tensor_in, n, indices=None):
    N, H, W, C = tf.convert_to_tensor(tensor_in).shape.as_list()
    S = H * W
    tensor_NSC = tf.reshape(tensor_in, [N, S, C])
    all_indices = list(range(S))
    shuffled_indices = tf.random_shuffle(all_indices)
    indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices
    res = tf.gather(tensor_NSC, indices, axis=1)
    return res, indices


def random_pooling(feats, output_1d_size=100):
    is_input_tensor = type(feats) is tf.Tensor

    if is_input_tensor:
        feats = [feats]

    # convert all inputs to tensors
    feats = [tf.convert_to_tensor(feats_i) for feats_i in feats]

    N, H, W, C = feats[0].shape.as_list()
    feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2)
    res = [feats_sampled_0]
    for i in range(1, len(feats)):
        feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
        res.append(feats_sampled_i)

    res = [tf.reshape(feats_sampled_i, [N, output_1d_size, output_1d_size, C]) for feats_sampled_i in res]
    if is_input_tensor:
        return res[0]
    return res


def crop_quarters(feature_tensor):
    N, fH, fW, fC = feature_tensor.shape.as_list()
    quarters_list = []
    quarter_size = [N, round(fH / 2), round(fW / 2), fC]
    quarters_list.append(tf.slice(feature_tensor, [0, 0, 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, 0, round(fW / 2), 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), round(fW / 2), 0], quarter_size))
    feature_tensor = tf.concat(quarters_list, axis=0)
    return feature_tensor


def id_mrf_reg_feat(feat_A, feat_B, config, args):
    if config.crop_quarters is True:
        feat_A = crop_quarters(feat_A)
        feat_B = crop_quarters(feat_B)

    N, fH, fW, fC = feat_A.shape.as_list()
    if fH * fW <= config.max_sampling_1d_size ** 2:
        print(' #### Skipping pooling ....')
    else:
        print(' #### pooling %d**2 out of %dx%d' % (config.max_sampling_1d_size, fH, fW))
        feat_A, feat_B = random_pooling([feat_A, feat_B], output_1d_size=config.max_sampling_1d_size)

    return mrf_loss(feat_A, feat_B, distance=config.Dist, nnsigma=config.nn_stretch_sigma, args=args)


from easydict import EasyDict as edict
# scale of im_src and im_dst: [-1, 1]
def grad_matching_loss(im_src, im_dst, config):

    match_config = edict()
    match_config.crop_quarters = False
    match_config.max_sampling_1d_size = 65
    match_config.Dist = Distance.DotProduct
    match_config.nn_stretch_sigma = 0.5  # 0.1

    match_loss = id_mrf_reg_feat(im_src, im_dst, match_config, config)

    match_loss = tf.reduce_sum(match_loss)

    return match_loss


"""
Salient Edge
"""
import cv2
def gaussian_kernel_2d_opencv(kernel_size = 3,sigma = 0):
    """
    ref: https://blog.csdn.net/qq_16013649/article/details/78784791
    ref: tensorflow
        (1) https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
        (2) https://github.com/tensorflow/tensorflow/issues/2826
    """
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky))

def priority_loss_mask(mask, ksize=5, sigma=1, iteration=2):
    gaussian_kernel = gaussian_kernel_2d_opencv(kernel_size=ksize, sigma=sigma)
    gaussian_kernel = np.reshape(gaussian_kernel, (ksize, ksize, 1, 1))
    mask_priority = tf.convert_to_tensor(mask, dtype=tf.float32)
    for i in range(iteration):
        mask_priority = tf.nn.conv2d(mask_priority, gaussian_kernel, strides=[1,1,1,1], padding='SAME')

    return mask_priority


# structure loss
from skimage import feature
from skimage.color import rgb2gray

"""
Structure loss
"""
import cv2

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