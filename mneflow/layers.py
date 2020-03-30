# -*- coding: utf-8 -*-
"""
Defines mneflow.layers for mneflow.models.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import functools
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from numpy import prod, sqrt


def compose(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def stack_layers(*args):
    return functools.partial(functools.reduce, compose)(*args)


def vgg_block(n_layers, layer, kwargs):
    layers = []
    for i in range(n_layers):
        if i > 0:
            kwargs['inch'] = kwargs['n_ls']
        layers.append(layer(**kwargs))
    layers.append(tf.layers.batch_normalization)
    layers.append(tf.nn.max_pool)
    return stack_layers(layers[::-1])


def weight_variable(shape, name='', method='he'):
    """Initialize weight variable."""
    if method == 'xavier':
        xavf = 2./sum(prod(shape[:-1]))
        initial = xavf*tf.random_uniform(shape, minval=-.5, maxval=.5)

    elif method == 'he':
        hef = sqrt(6. / prod(shape[:-1]))
        initial = hef*tf.random_uniform(shape, minval=-1., maxval=1.)

    else:
        initial = tf.truncated_normal(shape, stddev=.1)

    return tf.Variable(initial, trainable=True, name=name+'weights')


def bias_variable(shape):
    """Initialize bias variable as constant 0.1."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True, name='bias')


def spatial_dropout(x, rate, seed=1234):
    num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]
    random_tensor = 1 - rate
    random_tensor = random_tensor + tf.random_uniform(num_feature_maps,
                                                      seed=seed,
                                                      dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    binary_tensor = tf.reshape(binary_tensor, [-1, 1, 1, tf.shape(x)[3]])
    ret = tf.div(x, (1 - rate)) * binary_tensor
    return ret


# ----- Layers -----
class Dense():
    """Fully-connected layer."""
    def __init__(self, scope="fc", size=None, dropout=.5,
                 nonlin=tf.identity):
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = int(size)
        self.dropout = dropout
        self.nonlin = nonlin

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        with tf.name_scope(self.scope):
            # print(int(prod(x.get_shape().as_list()[1:])),
            #       type(int(prod(x.get_shape().as_list()[1:]))))
            while True:
                # reuse weights if already initialized
                try:
                    if len(x.shape) > 2:  # flatten if input is not 2d array
                        # print(self.flatsize)
                        x = tf.reshape(x, [-1, self.flatsize])
                    tmp = tf.matmul(x, self.w)
                    # print('matmul shape:', tmp.shape)
                    tmp = tmp + self.b
                    # print('added bias shape:', tmp.shape)
                    tmp = self.nonlin(tmp, name='out')
                    # print('after nonlin:', tmp.shape)
                    return tmp

                except(AttributeError):
                    if len(x.shape) > 2:
                        self.flatsize = int(prod(x.get_shape().as_list()[1:]))
                    else:
                        self.flatsize = x.get_shape().as_list()[1]

                    print(self.scope, ':::', self.flatsize, self.size)
                    self.w = weight_variable((self.flatsize, self.size),
                                             name='fc_')
                    self.b = bias_variable([self.size])
                    self.w = tf.nn.dropout(self.w, rate=self.dropout)
                    print(self.scope, 'init : OK')

class TempPooling():
    def __init__(self, scope="pool", stride=2, pooling=2,
                       padding='SAME', pool_type='max', **args):
        self.scope = '_'.join([pool_type, scope])
        self.strides = [1, 1, stride,  1]
        self.kernel = [1, 1, pooling,  1]
        self.padding = padding
        self.pool_type = pool_type

    def __call__(self, x):
        if self.pool_type == 'avg':
            pooled = tf.nn.avg_pool2d(
                                x,
                                ksize=self.kernel,
                                strides=self.strides,
                                padding=self.padding,
                                data_format='NHWC')
        else:
            pooled = tf.nn.max_pool2d(
                                x,
                                ksize=self.kernel,
                                strides=self.strides,
                                padding=self.padding,
                                data_format='NHWC')
        return pooled



class LFTConv():
    """Stackable temporal convolutional layer, interpretable (LF)."""
    def __init__(self, scope="lf-conv", n_ls=32,  nonlin=tf.nn.relu,
                 filter_length=7, padding='SAME',
                 #stride=1, pooling=2, pool_type='max',
                 **args):
        self.scope = scope
#        super(LFTConv, self).__init__(name=scope, **args)
        self.size = n_ls
        self.filter_length = filter_length
#        self.stride = stride
#        self.pooling = pooling
#        self.pool_type = pool_type
        self.nonlin = nonlin
        self.padding = padding

        #self.kernel_regularizer = kernel_regularizer
        #self.bias_regularizer = bias_regularizer
        ##############################################

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                # reuse weights if already initialized
                try:
                    # print('lf-inp', x.shape)
                    conv = tf.nn.depthwise_conv2d(x,
                                                  self.filters,
                                                  padding=self.padding,
                                                  strides=[1, 1, 1, 1],
                                                  data_format='NHWC')
                    conv = self.nonlin(conv + self.b)

                    #print('f:', self.filters.shape)
                    print('lf-out', conv.shape)
                    return conv
                except(AttributeError):
                    self.filters = weight_variable(
                            [1, self.filter_length, x.shape[-1].value, 1],
                            name='tconv_')
                    self.b = bias_variable([x.shape[-1].value])
                    print(self.scope, 'init : OK')


class VARConv():
    """Stackable spatio-temporal convolutional Layer (VAR)."""
    def __init__(self, scope="var-conv", n_ls=32,  nonlin=tf.nn.relu,
                 filter_length=7, padding='SAME',
                 #stride=1, pooling=2, pool_type='max',
                 **args):
        self.scope = scope
        self.size = n_ls
        self.filter_length = filter_length
#        self.stride = stride
#        self.pooling = pooling
#        self.pool_type = pool_type
        self.nonlin = nonlin
        self.padding = padding


    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                # reuse weights if already initialized
                try:
                    conv = tf.nn.conv2d(x,
                                        self.filters,
                                        padding=self.padding,
                                        strides=[1, 1, 1, 1],
                                        data_format='NHWC')
                    conv = self.nonlin(conv + self.b)
#                    conv = tf.nn.max_pool(conv, ksize=[1, self.pooling, 1, 1],
#                                          strides=[1, self.stride, 1, 1],
#                                          padding=self.padding)
                    if self.pool_type == 'avg':
                        conv = tf.nn.avg_pool2d(
                                conv,
                                ksize=[1, self.pooling, 1, 1],
                                strides=[1, self.stride, 1, 1],
                                padding=self.padding,
                                data_format='NHWC')
                    else:
                        conv = tf.nn.max_pool2d(
                                conv,
                                ksize=[1, self.pooling, 1, 1],
                                strides=[1, self.stride, 1, 1],
                                padding=self.padding,
                                data_format='NHWC')
                    print(self.scope, 'init:OK shape:', conv.shape)
                    return conv

                except(AttributeError):
                    self.filters = weight_variable(
                            [self.filter_length, 1, x.shape[-1].value,
                             self.size],
                            name='tconv_')
                    self.b = bias_variable([self.size])
                    print(self.scope, 'init')


class DeMixing():
    """Reduce dimensions across one domain."""
    def __init__(self, scope="de-mix", n_ls=32,  nonlin=tf.identity, axis=3):
        self.scope = scope
        self.size = n_ls
        self.nonlin = nonlin
        self.axis = axis

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                # reuse weights if already initialized
                try:
                    x_reduced = self.nonlin(
                            tf.tensordot(x, self.W, axes=[[self.axis], [0]],
                                         name='de-mix')
                            + self.b_in)
#                    if self.axis == 2:
#                        x_reduced = tf.transpose(x_reduced, perm = [0,1,3,2])

                    print('dmx', x_reduced.shape)
                    return x_reduced
                except(AttributeError):
                    self.W = weight_variable(
                            (x.shape[self.axis].value, self.size), name='dmx_')
                    self.b_in = bias_variable([self.size])
                    print(self.scope, 'init : OK')


class ConvDSV():
    """Standard/Depthwise/Spearable Convolutional Layer constructor."""

    def __init__(self, scope="conv", n_ls=None, nonlin=None, inch=None,
                 domain=None, padding='SAME', filter_length=5, stride=1,
                 pooling=2, dropout=.5, conv_type='depthwise'):

        self.scope = '-'.join([conv_type, scope, domain])
        self.padding = padding
        self.domain = domain
        self.inch = inch
        self.dropout = dropout
        self.size = n_ls
        self.filter_length = filter_length
        self.stride = stride
        self.nonlin = nonlin
        self.conv_type = conv_type

    def __call__(self, x):
        """Calculate the graph for input `X`.

        Raises:
        -------
            ValueError: If the convolution/domain arguments do not have
            the supported values.
        """
        with tf.name_scope(self.scope):
            while True:
                try:
                    if self.conv_type == 'depthwise':
                        conv_ = tf.nn.depthwise_conv2d(
                                x,
                                self.filters,
                                strides=[1, self.stride, 1, 1],
                                padding=self.padding)

                    elif self.conv_type == 'separable':
                        conv_ = tf.nn.separable_conv2d(
                                x,
                                self.filters,
                                self.pwf,
                                strides=[1, self.stride, 1, 1],
                                padding=self.padding)

                    elif self.conv_type == '2d':
                        conv_ = tf.nn.conv2d(
                                x,
                                self.filters,
                                strides=[1, self.stride, self.stride, 1],
                                padding=self.padding)
                    else:
                        raise ValueError('Invalid convolution type.')

                    conv_ = self.nonlin(conv_ + self.b)

                    return conv_

                except(AttributeError):
                    if self.domain == 'time':
                        w_sh = [1, self.filter_length, self.inch, self.size]

                    elif self.domain == 'space':
                        w_sh = [self.filter_length, 1, self.inch, self.size]

                    elif self.domain == '2d':
                        w_sh = [self.filter_length[0], self.filter_length[1],
                                self.inch, self.size]
                    else:
                        raise ValueError('Invalid domain.')

                    self.filters = weight_variable(w_sh, name='weights')
                    self.b = bias_variable([self.size])

                    if self.conv_type == 'separable':
                        self.pwf = weight_variable(
                                [1, 1, self.inch*self.size, self.size],
                                name='sep-pwf')

                    print(self.scope, 'init : OK')


