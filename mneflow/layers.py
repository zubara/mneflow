# -*- coding: utf-8 -*-
"""
Defines mneflow.layers for mneflow.models.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
#TODO: keras compatible layers
#TODO: pooling layer

#import functools
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import relu
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np

class BaseLayer(tf.keras.layers.Layer):
    def __init__(self, size, nonlin, specs, **args):
        super(BaseLayer, self).__init__(**args)
        self.size = size
        self.nonlin = nonlin
        self.specs = specs
        self.specs.setdefault("dropout", 0.)
        self.specs.setdefault("l1", 0.)
        self.specs.setdefault("l2", 0.)
        self.specs.setdefault("kernel_regularizer", "l2")
        self.specs.setdefault("bias_regularizer", "l1")
        # self.specs.setdefault("kernel_constraint", "l2")

        print(self.scope, 'init : OK')

    def call(self, x):
        while True:
            with tf.name_scope(self.scope):
                try:
                    print("BaseLayer: called")
                    return x
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)
                    print(self.scope, 'building from call : OK')

    def build(self, input_shape):
        print("Called build")


class Dense(BaseLayer, tf.keras.layers.Layer):
    """
    Fully-connected layer
    """
    def __init__(self, scope="fc", size=None, nonlin=tf.identity, specs={},
                 **args):
        self.scope = scope
        super(Dense, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)
        self.dropout = self.specs['dropout']
#        print(self.scope, 'init : OK')

    def get_config(self):
        config = self.get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'dropout': self.dropout, 'nonlin': self.nonlin})
        return config

    def build(self, input_shape):
        #super(Dense, self).build(input_shape)
        # print(input_shape)
        self.flatsize = np.prod(input_shape[1:]).value
        print(self.scope, ':::', self.flatsize, self.size)

        self.w = self.add_weight(shape=[self.flatsize, self.size],
                                 initializer='he_uniform',
                                 #regularizer=self.kernel_regularizer,
                                 trainable=True,
                                 name='fc_weights',
                                 dtype=tf.float32)

        self.b = self.add_weight(shape=[self.size],
                                 initializer=Constant(0.1),
                                 #regularizer=self.bias_regularizer,
                                 trainable=True,
                                 name='fc_bias',
                                 dtype=tf.float32)

        print(self.scope, 'build : OK')


    def call(self, x, training=None):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        while True:
            with tf.name_scope(self.scope):
                try:
                    if len(x.shape) > 2:  # flatten if input is not 2d array
                        # print(self.flatsize)
                        x = tf.reshape(x, [-1, self.flatsize])
                    rate = self.dropout if training else 0.0
                    tmpw = tf.nn.dropout(self.w, rate=rate)

                    tmp = tf.matmul(x, tmpw) + self.b
                    tmp = self.nonlin(tmp, name='out')
                    return tmp
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)
                    print(self.scope, 'building from call')


class DeMixing(BaseLayer):
    """
    Spatial demixing Layer
    """
    def __init__(self, scope="de-mix", size=None, nonlin=tf.identity, axis=-1,
                 specs={},
                 **args):
        self.scope = scope
        super(DeMixing, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)
        self.axis = axis
#        print(self.scope, 'init : OK')

    def get_config(self):
        config = super(DeMixing, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'nonlin': self.nonlin, 'axis': self.axis})
        return config

    def build(self, input_shape):

        #super(DeMixing, self).build(input_shape)
        self.W = self.add_weight(
                shape=(input_shape[self.axis].value, self.size),
                initializer='he_uniform',
                regularizer=self.specs['kernel_regularizer'],
                trainable=True,
                name='dmx_weights',
                dtype=tf.float32)

        self.b_in = self.add_weight(shape=([self.size]),
                                    initializer=Constant(0.1),
                                    regularizer=self.specs['bias_regularizer'],
                                    trainable=True,
                                    name='bias',
                                    dtype=tf.float32)
        #print(self.scope, 'built : OK')

    def call(self, x, training=None):
        while True:
            with tf.name_scope(self.scope):
                try:
                    demix = tf.tensordot(x, self.W, axes=[[self.axis], [0]],
                                         name='de-mix')
                    demix = self.nonlin(demix + self.b_in)
                    print('dmx', demix.shape)
                    return demix
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)
                    print(self.scope, 'building from call')


class LFTConv(BaseLayer):
    """
    Stackable temporal convolutional layer, interpreatble (LF)
    """
    def __init__(self, scope="lf-conv", size=32,  nonlin=tf.nn.relu,
                 filter_length=7, pooling=2, padding='SAME', specs={},
                 **args):
        self.scope = scope
        super(LFTConv, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)
        self.size = size
        self.filter_length = filter_length
        self.padding = padding

    def get_config(self):

        config = super(LFTConv, self).get_config()
        config.update({'scope': self.scope,
                       'filter_length': self.filter_length,
                       'nonlin': self.nonlin, 'padding': self.padding})
        return config

    def build(self, input_shape):
        #super(LFTConv, self).build(input_shape)
        shape = [1, self.filter_length, input_shape[-1].value, 1]
        self.filters = self.add_weight(shape=shape,
                                       initializer='he_uniform',
                                       regularizer=self.specs['kernel_regularizer'],
                                       trainable=True,
                                       name='tconv_weights',
                                       dtype=tf.float32)

        self.b = self.add_weight(shape=([input_shape[-1].value]),
                                 initializer=Constant(0.1),
                                 regularizer=self.specs['bias_regularizer'],
                                 trainable=True,
                                 name='bias',
                                 dtype=tf.float32)
        print(self.scope, 'build : OK')

    def call(self, x, training=None):
        while True:
            with tf.name_scope(self.scope):
                try:
                    conv = tf.nn.depthwise_conv2d(x,
                                                  self.filters,
                                                  padding=self.padding,
                                                  strides=[1, 1, 1, 1],
                                                  data_format='NHWC')
                    conv = self.nonlin(conv + self.b)
                    print('lf-out', conv.shape)
                    return conv
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)
                    print(self.scope, 'building from call')


#def compose(f, g):
#    return lambda *a, **kw: f(g(*a, **kw))
#
#
#def stack_layers(*args):
#    return functools.partial(functools.reduce, compose)(*args)
#
#
#def vgg_block(n_layers, layer, kwargs):
#    layers = []
#    for i in range(n_layers):
#        if i > 0:
#            kwargs['inch'] = kwargs['n_ls']
#        layers.append(layer(**kwargs))
#    layers.append(tf.layers.batch_normalization)
#    layers.append(tf.nn.max_pool)
#    return stack_layers(layers[::-1])


def weight_variable(shape, name='', method='he'):
    """Initialize weight variable."""
    if method == 'xavier':
        xavf = 2./sum(np.prod(shape[:-1]))
        initial = xavf*tf.random_uniform(shape, minval=-.5, maxval=.5)

    elif method == 'he':
        hef = np.sqrt(6. / np.prod(shape[:-1]))
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
#class Dense():
#    """Fully-connected layer."""
#    def __init__(self, scope="fc", size=None, dropout=.5,
#                 nonlin=tf.identity):
#        assert size, "Must specify layer size (num nodes)"
#        self.scope = scope
#        self.size = int(size)
#        self.dropout = dropout
#        self.nonlin = nonlin
#
#    def __call__(self, x):
#        """Dense layer currying, to apply layer to any input tensor `x`"""
#        with tf.name_scope(self.scope):
#            # print(int(prod(x.get_shape().as_list()[1:])),
#            #       type(int(prod(x.get_shape().as_list()[1:]))))
#            while True:
#                # reuse weights if already initialized
#                try:
#                    if len(x.shape) > 2:  # flatten if input is not 2d array
#                        # print(self.flatsize)
#                        x = tf.reshape(x, [-1, self.flatsize])
#                    tmp = tf.matmul(x, self.w)
#                    # print('matmul shape:', tmp.shape)
#                    tmp = tmp + self.b
#                    # print('added bias shape:', tmp.shape)
#                    tmp = self.nonlin(tmp, name='out')
#                    # print('after nonlin:', tmp.shape)
#                    return tmp
#
#                except(AttributeError):
#                    if len(x.shape) > 2:
#                        self.flatsize = int(np.prod(x.get_shape().as_list()[1:]))
#                    else:
#                        self.flatsize = x.get_shape().as_list()[1]
#
#                    print(self.scope, ':::', self.flatsize, self.size)
#                    self.w = weight_variable((self.flatsize, self.size),
#                                             name='fc_')
#                    self.b = bias_variable([self.size])
#                    self.w = tf.nn.dropout(self.w, rate=self.dropout)
#                    print(self.scope, 'init : OK')

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
        print(self.scope, ":", pooled.shape)
        return pooled



#class LFTConv():
#    """Stackable temporal convolutional layer, interpretable (LF)."""
#    def __init__(self, scope="lf-conv", n_ls=32,  nonlin=tf.nn.relu,
#                 filter_length=7, padding='SAME',
#                 #stride=1, pooling=2, pool_type='max',
#                 **args):
#        self.scope = scope
##        super(LFTConv, self).__init__(name=scope, **args)
#        self.size = n_ls
#        self.filter_length = filter_length
#        self.nonlin = nonlin
#        self.padding = padding
#
#        #self.kernel_regularizer = kernel_regularizer
#        #self.bias_regularizer = bias_regularizer
#        ##############################################
#
#    def __call__(self, x):
#        with tf.name_scope(self.scope):
#            while True:
#                # reuse weights if already initialized
#                try:
#                    # print('lf-inp', x.shape)
#                    conv = tf.nn.depthwise_conv2d(x,
#                                                  self.filters,
#                                                  padding=self.padding,
#                                                  strides=[1, 1, 1, 1],
#                                                  data_format='NHWC')
#                    conv = self.nonlin(conv + self.b)
#
#                    #print('f:', self.filters.shape)
#                    print('lf-out', conv.shape)
#                    return conv
#                except(AttributeError):
#                    self.filters = weight_variable(
#                            [1, self.filter_length, x.shape[-1].value, 1],
#                            name='tconv_')
#                    self.b = bias_variable([x.shape[-1].value])
#                    print(self.scope, 'init : OK')


class VARConv(tf.keras.layers.Layer):
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


#class DeMixing():
#    """Reduce dimensions across one domain."""
#    def __init__(self, scope="de-mix", n_ls=32,  nonlin=tf.identity, axis=3):
#        self.scope = scope
#        self.size = n_ls
#        self.nonlin = nonlin
#        self.axis = axis
#
#    def __call__(self, x):
#        with tf.name_scope(self.scope):
#            while True:
#                # reuse weights if already initialized
#                try:
#                    x_reduced = self.nonlin(
#                            tf.tensordot(x, self.W, axes=[[self.axis], [0]],
#                                         name='de-mix')
#                            + self.b_in)
#                    print('dmx', x_reduced.shape)
#                    return x_reduced
#                except(AttributeError):
#                    self.W = weight_variable(
#                            (x.shape[self.axis].value, self.size), name='dmx_')
#                    self.b_in = bias_variable([self.size])
#                    print(self.scope, 'init : OK')


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


