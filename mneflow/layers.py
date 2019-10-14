# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:40:09 2017

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import tensorflow as tf
from numpy import prod, sqrt
import functools


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


class Dense():
    """
    Fully-connected layer
    """
    def __init__(self, scope="fc", size=None, dropout=.5,
                 nonlin=tf.identity):
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout
        self.nonlin = nonlin

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        with tf.name_scope(self.scope):
            while True:
                try:  # reuse weights if already initialized
                    if len(x.shape) > 2:  # flatten if input is not 2d array
                        x = tf.reshape(x, [-1, self.flatsize])
                    return self.nonlin(tf.matmul(x, self.w) + self.b,
                                       name='out')
                except(AttributeError):
                    if len(x.shape) > 2:
                        self.flatsize = prod(x.shape[1:]).value
                    else:
                        self.flatsize = x.shape[1].value
                    print(self.scope, self.flatsize)
                    self.w = weight_variable((self.flatsize, self.size),
                                             name='fc_')
                    self.b = bias_variable([self.size])
                    self.w = tf.nn.dropout(self.w, self.dropout)
                    print(self.scope, 'init : OK')


class LFTConv():
    """
    Stackable temporal convolutional layer, interpreatble (LF)
    """
    def __init__(self, scope="lf-conv", n_ls=32,  nonlin=tf.nn.relu,
                 filter_length=7, stride=1, pooling=2, padding='SAME'):
        self.scope = scope
        self.size = n_ls
        self.filter_length = filter_length
        self.stride = stride
        self.pooling = pooling
        self.nonlin = nonlin
        self.padding = padding

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                try:  # reuse weights if already initialized
                    conv = tf.nn.depthwise_conv2d(x, self.filters,
                                                  padding=self.padding,
                                                  strides=[1, 1, 1, 1],
                                                  data_format='NHWC')
                    conv = self.nonlin(conv + self.b)
                    conv = tf.nn.max_pool(conv, ksize=[1, self.pooling, 1, 1],
                                          strides=[1, self.stride, 1, 1],
                                          padding=self.padding)
                    return conv
                except(AttributeError):
                    self.filters = weight_variable([self.filter_length, 1, self.size, 1],
                                                   name='tconv_')
                    self.b = bias_variable([self.size])
                    print(self.scope, 'init : OK')


class VARConv():
    """
    Stackable spatio-temporal convolutional Layer (VAR)
    """
    def __init__(self, scope="var-conv", n_ls=32,  nonlin=tf.nn.relu,
                 filter_length=7, stride=1, pooling=2, padding='SAME'):
        self.scope = scope
        self.size = n_ls
        self.filter_length = filter_length
        self.stride = stride
        self.pooling = pooling
        self.nonlin = nonlin
        self.padding = padding

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                try:  # reuse weights if already initialized
                    conv = tf.nn.conv2d(x, self.filters, padding=self.padding,
                                        strides=[1, 1, 1, 1],
                                        data_format='NHWC')
                    conv = self.nonlin(conv + self.b)
#                    conv = tf.nn.max_pool(conv, ksize=[1, self.pooling, 1, 1],
#                                          strides=[1, self.stride, 1, 1],
#                                          padding=self.padding)
                    conv = tf.nn.max_pool(conv, ksize=[1, self.pooling, 1, 1],
                                          strides=[1, self.stride, 1, 1],
                                          padding='VALID')
                    print(conv.shape)
                    return conv
                except(AttributeError):
                    self.filters = weight_variable([self.filter_length, 1,
                                                    x.shape[-1].value,
                                                    self.size],
                                                   name='tconv_')
                    self.b = bias_variable([self.size])
                    print(self.scope, 'init : OK')


class DeMixing():
    """
    Spatial demixing Layer
    """
    def __init__(self, scope="de-mix", n_ls=32,  nonlin=tf.identity):
        self.scope = scope
        self.size = n_ls
        self.nonlin = nonlin

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                try:  # reuse weights if already initialized
                    x_reduced = self.nonlin(tf.tensordot(x, self.W,
                                                         axes=[[1], [0]],
                                                         name='de-mix') +
                                            self.b_in)
                    x_reduced = tf.expand_dims(x_reduced, -2)
                    return x_reduced
                except(AttributeError):
                    self.W = weight_variable((x.shape[1].value, self.size),
                                             name='dmx_')
                    self.b_in = bias_variable([self.size])
                    print(self.scope, 'init : OK')


def spatial_dropout(x, keep_prob, seed=1234):
    num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(num_feature_maps,
                                       seed=seed,
                                       dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    binary_tensor = tf.reshape(binary_tensor,
                               [-1, 1, 1, tf.shape(x)[3]])
    ret = tf.div(x, keep_prob) * binary_tensor
    return ret


class ConvDSV():
    """
    Standard/Depthwise/Spearable Convolutional Layer constructor
    """

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
        self.pool = pooling
        self.nonlin = nonlin
        self.conv_type = conv_type

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                try:
                    if self.conv_type == 'depthwise':
                        conv_ = self.nonlin(tf.nn.depthwise_conv2d(x,
                                            self.filters,
                                            strides=[1, self.stride, 1, 1],
                                            padding=self.padding) + self.b)

                    elif self.conv_type == 'separable':
                        conv_ = self.nonlin(tf.nn.separable_conv2d(x,
                                            self.filters, self.pwf,
                                            strides=[1, self.stride, 1, 1],
                                            padding=self.padding) + self.b)

                    elif self.conv_type == '2d':
                        conv_ = self.nonlin(tf.nn.conv2d(x, self.filters,
                                            strides=[1, self.stride, self.stride, 1],
                                            padding=self.padding) + self.b)

                    conv_ = tf.nn.max_pool(conv_, ksize=[1, self.pool, 1, 1],
                                           strides=[1, 1, 1, 1],
                                           padding='SAME')
                    return conv_

                except(AttributeError):
                    if self.domain == 'time':
                        self.filters = weight_variable([1, self.filter_length,
                                                       self.inch, self.size],
                                                       name='weights')

                    elif self.domain == 'space':
                        self.filters = weight_variable([self.filter_length, 1,
                                                       self.inch, self.size],
                                                       name='weights')
                    elif self.domain == '2d':
                        self.filters = weight_variable([self.filter_length[0],
                                                       self.filter_length[1],
                                                       self.inch, self.size],
                                                       name='weights')
                    self.b = bias_variable([self.size])

                    if self.conv_type == 'separable':
                        self.pwf = weight_variable([1, 1, self.inch*self.size,
                                                    self.size], name='sep-pwf')
                    print(self.scope, 'init : OK')


def weight_variable(shape, name='', method='he'):
    #    """Initialize weight variable"""
    if method == 'xavier':
        xavf = 2/sum(prod(shape[:-1]))
        initial = xavf*tf.random_uniform(shape, minval=-.5, maxval=.5)
    elif method == 'he':
        hef = sqrt(6. / prod(shape[:-1]))
        initial = hef*tf.random_uniform(shape, minval=-1., maxval=1.)
    else:
        initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial, trainable=True, name=name+'weights')


def bias_variable(shape):
    #    """ Initialize bias variable as constant 0.1"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True, name='bias')


class DeConvLayer():
    """DeConvolution Layer"""
    def __init__(self, n_ls, y_shape, scope="deconv", flat_out=False,
                 filter_length=5):
        self.scope = scope
        self.n_ch, self.n_t = y_shape
        self.size = n_ls
        self.filter_length = filter_length
        self.flat_out = flat_out

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                try:
                    latent = tf.nn.relu(tf.tensordot(x, self.W,
                                                     axes=[[1], [0]]) +
                                        self.b_in)

                    print('x_reduced', latent.shape)
                    x_perm = tf.expand_dims(latent, 1)
                    print('x_perm', x_perm.shape)

                    conv_ = tf.einsum('lij, ki -> lkj', x_perm, self.filters)
                    print('deconv:', conv_.shape)
                    out = tf.einsum('lkj, jm -> lmk', conv_, self.demixing)
                    out = out + self.b_out
                    if self.flat_out:
                        return tf.reshape(out, [-1, self.n_t*self.n_ch])
                    else:
                        print(out.shape)
                        return out
                except(AttributeError):
                    self.W = weight_variable((x.get_shape()[1].value,
                                              self.size))
                    self.W = tf.nn.dropout(self.W, rate=.5)
                    self.b_in = bias_variable([self.size])
                    self.filters = weight_variable([self.n_t, 1])
                    #  self.b = bias_variable([self.size])
                    self.demixing = weight_variable((self.size, self.n_ch))
                    self.demixing = tf.nn.dropout(self.demixing, rate=.5)
                    self.b_out = bias_variable([self.n_ch])
                    print(self.scope, 'init : OK')
