#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mneflow.Layers implementation using keras layers

@author: vranoug1
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
# from tensorflow.compat.v1.keras.initializers import he_uniform
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import relu

# TODO: add vgg_block layer


class Dense(layers.Layer):
    """
    Fully-connected layer
    """
    def __init__(self, scope="fc", size=None, dropout=.5, nonlin=tf.identity,
                 kernel_regularizer=None, bias_regularizer=None, **args):
        assert size, "Must specify layer size (num nodes)"
        super(Dense, self).__init__(name=scope, **args)
        self.scope = scope
        self.size = size
        self.dropout = dropout
        self.nonlin = nonlin
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        print(self.scope, 'init : OK')

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'dropout': self.dropout, 'nonlin': self.nonlin})
        return config

    def build(self, input_shape):
        super(Dense, self).build(input_shape)
        if len(input_shape) > 2:
            self.flatsize = np.prod(input_shape[1:]).value
        else:
            self.flatsize = input_shape[1].value

        self.w = self.add_weight(shape=(self.flatsize, self.size),
                                 initializer='he_uniform',
                                 regularizer=self.kernel_regularizer,
                                 trainable=True,
                                 name='fc_weights',
                                 dtype=tf.float32)

        self.b = self.add_weight(shape=([self.size]),
                                 initializer=Constant(0.1),
                                 regularizer=self.bias_regularizer,
                                 trainable=True,
                                 name='bias',
                                 dtype=tf.float32)

        self.w = tf.nn.dropout(self.w, rate=self.dropout)
        print(self.scope, 'build : OK')

    def call(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        with tf.name_scope(self.scope):
            if len(x.shape) > 2:  # flatten if input is not 2d array
                x = tf.reshape(x, [-1, self.flatsize])
            out = self.nonlin(tf.matmul(x, self.w) + self.b, name='out')
            return out


class LFTConv(layers.Layer):
    """
    Stackable temporal convolutional layer, interpreatble (LF)
    """
    def __init__(self, scope="lf-conv", n_ls=32,  nonlin=tf.nn.relu,
                 filter_length=7, stride=1, pooling=2, padding='SAME',
                 kernel_regularizer=None, bias_regularizer=None, **args):
        super(LFTConv, self).__init__(name=scope, **args)
        self.scope = scope
        self.size = n_ls
        self.filter_length = filter_length
        self.stride = stride
        self.pooling = pooling
        self.nonlin = nonlin
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def get_config(self):

        config = super(LFTConv, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'filter_length': self.filter_length,
                       'stride': self.stride, 'pooling': self.pooling,
                       'nonlin': self.nonlin, 'padding': self.padding})
        return config

    def build(self, input_shape):
        super(LFTConv, self).build(input_shape)
        shape = [self.filter_length, 1, self.size, 1]
        self.filters = self.add_weight(shape=shape,
                                       initializer='he_uniform',
                                       regularizer=self.kernel_regularizer,
                                       trainable=True,
                                       name='tconv_weights',
                                       dtype=tf.float32)

        self.b = self.add_weight(shape=([self.size]),
                                 initializer=Constant(0.1),
                                 regularizer=self.bias_regularizer,
                                 trainable=True,
                                 name='bias',
                                 dtype=tf.float32)
        print(self.scope, 'build : OK')

    def call(self, x):
        with tf.name_scope(self.scope):
            conv = tf.nn.depthwise_conv2d(x,
                                          self.filters,
                                          padding=self.padding,
                                          strides=[1, 1, 1, 1],
                                          data_format='NHWC')
            conv = self.nonlin(conv + self.b)
            conv = tf.nn.max_pool(conv,
                                  ksize=[1, self.pooling, 1, 1],
                                  strides=[1, self.stride, 1, 1],
                                  padding=self.padding)
            conv = tf.squeeze(conv, axis=2)
            #print(self.scope, conv.shape)
            return conv


class VARConv(layers.Layer):
    """
    Stackable spatio-temporal convolutional Layer (VAR)
    """
    def __init__(self, scope="var-conv", n_ls=32, nonlin=relu,
                 filter_length=7, stride=1, pooling=2, padding='SAME',
                 kernel_regularizer=None, bias_regularizer=None, **args):
        super(VARConv, self).__init__(name=scope, **args)
        self.scope = scope
        self.size = n_ls
        self.filter_length = filter_length
        self.stride = stride
        self.pooling = pooling
        self.nonlin = nonlin
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        print(self.scope, 'init : OK')

    def get_config(self):
        config = super(VARConv, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'filter_length': self.filter_length,
                       'stride': self.stride, 'pooling': self.pooling,
                       'nonlin': self.nonlin, 'padding': self.padding})
        return config

    def build(self, input_shape):
        super(VARConv, self).build(input_shape)
        shape = [self.filter_length, 1, input_shape[-1].value, self.size]
        self.filters = self.add_weight(shape=shape,
                                       initializer='he_uniform',
                                       regularizer=self.kernel_regularizer,
                                       trainable=True,
                                       name='tconv_weights',
                                       dtype=tf.float32)

        self.b = self.add_weight(shape=([self.size]),
                                 initializer=Constant(0.1),
                                 regularizer=self.bias_regularizer,
                                 trainable=True,
                                 name='bias',
                                 dtype=tf.float32)
        print(self.scope, 'build : OK')

    def call(self, x):
        with tf.name_scope(self.scope):
            conv = tf.nn.conv2d(x,
                                self.filters,
                                padding=self.padding,
                                strides=[1, 1, 1, 1],
                                data_format='NHWC')
            conv = self.nonlin(conv + self.b)

            conv = tf.nn.max_pool(conv,
                                  ksize=[1, self.pooling, 1, 1],
                                  strides=[1, self.stride, 1, 1],
                                  padding='VALID')
            conv = tf.squeeze(conv, axis=2)
            #print(self.scope, conv.shape)
            # # print(self.scope, 'call : OK')
            return conv


class DeMixing(layers.Layer):
    """
    Spatial demixing Layer
    """
    def __init__(self, scope="de-mix", n_ls=32,  axis=1, nonlin=tf.identity,
                 kernel_regularizer=None, bias_regularizer=None, **args):
        super(DeMixing, self).__init__(name=scope, **args)
        self.scope = scope
        self.size = n_ls
        self.nonlin = nonlin
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.axis = axis

        print(self.scope, 'init : OK')

    def get_config(self):
        config = super(DeMixing, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'nonlin': self.nonlin})
        return config

    def build(self, input_shape):
        super(DeMixing, self).build(input_shape)
        self.W = self.add_weight(shape=(input_shape[self.axis].value, self.size),
                                 initializer='he_uniform',
                                 regularizer=self.kernel_regularizer,
                                 trainable=True,
                                 name='dmx_weights',
                                 dtype=tf.float32)

        self.b_in = self.add_weight(shape=([self.size]),
                                    initializer=Constant(0.1),
                                    regularizer=self.bias_regularizer,
                                    trainable=True,
                                    name='bias',
                                    dtype=tf.float32)
        print(self.scope, 'built : OK')

    def call(self, x):
        with tf.name_scope(self.scope):
            demix = tf.tensordot(x, self.W, axes=[[self.axis], [0]], name='de-mix')
            demix = self.nonlin(demix + self.b_in)
            x_reduced = tf.expand_dims(demix, -2)

            #print(self.scope, x_reduced.shape)
            return x_reduced


class ConvDSV(layers.Layer):
    """
    Standard/Depthwise/Spearable Convolutional Layer constructor
    """

    def __init__(self, scope="conv", n_ls=None, nonlin=None, inch=None,
                 domain=None, padding='SAME', filter_length=5, stride=1,
                 pooling=2, dropout=.5, conv_type='depthwise',
                 kernel_regularizer=None, bias_regularizer=None, **args):

        assert domain in ['time', 'space', '2d'], "Unknown domain."
        assert conv_type in ['depthwise', 'separable', '2d'], "Unknown conv."

        tmp = '-'.join([conv_type, scope, domain])
        super(ConvDSV, self).__init__(name=tmp, **args)

        self.scope = tmp
        self.size = n_ls
        self.filter_length = filter_length
        self.stride = stride
        self.pool = pooling
        self.nonlin = nonlin
        self.padding = padding
        self.domain = domain
        self.inch = inch
        self.dropout = dropout
        self.conv_type = conv_type
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def get_config(self):
        config = super(ConvDSV, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'filter_length': self.filter_length,
                       'stride': self.stride, 'pool': self.pool,
                       'nonlin': self.nonlin, 'padding': self.padding,
                       'domain': self.domain, 'inch': self.inch,
                       'dropout': self.dropout, 'conv_type': self.conv_type})
        return config

    def build(self, input_shape):
        super(ConvDSV, self).build(input_shape)
        shape = None
        if self.domain == 'time':
            shape = [1, self.filter_length, self.inch, self.size]

        elif self.domain == 'space':
            shape = [self.filter_length, 1, self.inch, self.size]

        elif self.domain == '2d':
            shape = [self.filter_length[0], self.filter_length[1], self.inch, self.size]

        self.filters = self.add_weight(shape=shape,
                                       initializer='he_uniform',
                                       regularizer=self.kernel_regularizer,
                                       trainable=True,
                                       name='weights',
                                       dtype=tf.float32)

        self.b = self.add_weight(shape=([self.size]),
                                 initializer=Constant(0.1),
                                 regularizer=self.bias_regularizer,
                                 trainable=True,
                                 name='bias',
                                 dtype=tf.float32)

        if self.conv_type == 'separable':
            shape = [1, 1, self.inch*self.size, self.size]
            self.pwf = self.add_weight(shape=shape,
                                       initializer='he_uniform',
                                       regularizer=self.kernel_regularizer,
                                       trainable=True,
                                       name='sep-pwf',
                                       dtype=tf.float32)
        print(self.scope, 'build : OK')

    def call(self, x):
        with tf.name_scope(self.scope):
            conv_ = None

            if self.conv_type == 'depthwise':
                conv_ = tf.nn.depthwise_conv2d(x,
                                               self.filters,
                                               strides=[1, self.stride, 1, 1],
                                               padding=self.padding)

            elif self.conv_type == 'separable':
                conv_ = tf.nn.separable_conv2d(x,
                                               self.filters,
                                               self.pwf,
                                               strides=[1, self.stride, 1, 1],
                                               padding=self.padding)

            elif self.conv_type == '2d':
                conv_ = tf.nn.conv2d(x,
                                     self.filters,
                                     strides=[1, self.stride, self.stride, 1],
                                     padding=self.padding)

            conv_ = self.nonlin(conv_ + self.b)

            conv_ = tf.nn.max_pool(conv_,
                                   ksize=[1, self.pool, 1, 1],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            return conv_


class DeConvLayer(layers.Layer):
    # Migrated from dev.py
    """DeConvolution Layer"""
    def __init__(self, n_ls, n_ch, n_t, scope="deconv", flat_out=False,
                 filter_length=5, kernel_regularizer=None,
                 bias_regularizer=None, **args):
        super(DeConvLayer, self).__init__(name=scope, **args)
        self.scope = scope
        self.n_ch = n_ch
        self.n_t = n_t
        self.size = n_ls
        self.filter_length = filter_length
        self.flat_out = flat_out
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        print(self.scope, 'init : OK')

    def get_config(self):
        config = super(DeConvLayer, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'n_ch': self.n_ch, 'n_t': self.n_t,
                       'filter_length': self.filter_length,
                       'flat_out': self.flat_out})
        return config

    def build(self, input_shape):
        super(DeConvLayer, self).build(input_shape)
        self.W = self.add_weight(shape=(input_shape[1].value, self.size),
                                 initializer='he_uniform',
                                 regularizer=self.kernel_regularizer,
                                 trainable=True,
                                 name='conv_weights',
                                 dtype=tf.float32)

        self.filters = self.add_weight(shape=([self.n_t, 1]),
                                       initializer='he_uniform',
                                       regularizer=self.kernel_regularizer,
                                       trainable=True,
                                       name='filter_weights',
                                       dtype=tf.float32)

        self.b_in = self.add_weight(shape=([self.size]),
                                    initializer=Constant(0.1),
                                    regularizer=self.bias_regularizer,
                                    trainable=True,
                                    name='bias_in',
                                    dtype=tf.float32)

        self.b = self.add_weight(shape=([self.size]),
                                 initializer=Constant(0.1),
                                 regularizer=self.bias_regularizer,
                                 trainable=True,
                                 name='bias',
                                 dtype=tf.float32)

        self.b_out = self.add_weight(shape=([self.n_ch]),
                                     initializer=Constant(0.1),
                                     regularizer=self.bias_regularizer,
                                     trainable=True,
                                     name='bias_out',
                                     dtype=tf.float32)

        self.demixing = self.add_weight(shape=(self.size, self.n_ch),
                                        initializer='he_uniform',
                                        regularizer=self.kernel_regularizer,
                                        trainable=True,
                                        name='dmx_weights',
                                        dtype=tf.float32)

        print(self.scope, 'build : OK')

    def call(self, x):
        tmp = tf.tensordot(x, self.W, axes=[[1], [0]])
        latent = tf.nn.relu(tmp + self.b_in)

        #print('x_reduced', latent.shape)
        x_perm = tf.expand_dims(latent, 1)
        print('x_perm', x_perm.shape)

        conv_ = tf.einsum('lij, ki -> lkj', x_perm, self.filters)
        #print('deconv:', conv_.shape)
        out = tf.einsum('lkj, jm -> lmk', conv_, self.demixing)
        out = out
        if self.flat_out:
            return tf.reshape(out, [-1, self.n_t*self.n_ch])
        else:
            print(out.shape)
            return out


class LSTMv1(layers.LSTM):
    def __init__(self, scope="lstm", size=32, dropout=0.5, nonlin='tanh',
                 unit_forget_bias=True, kernel_regularizer=None,
                 bias_regularizer=None, return_sequences=True, **args):
        super(LSTMv1, self).__init__(name=scope, units=size,
                                     activation=nonlin,
                                     recurrent_activation=tf.nn.sigmoid,
                                     unit_forget_bias=unit_forget_bias,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     return_sequences=return_sequences, **args)
        self.scope = scope
        self.size = size
        self.nonlin = nonlin
        print(self.scope, 'init : OK')

    def get_config(self):
        config = super(LSTMv1, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'nonlin': self.nonlin})
        return config

    def build(self, input_shape):
        #print(self.scope, 'build : OK')
        super(LSTMv1, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        #print(self.scope, inputs.shape)
        return super(LSTMv1, self).call(inputs, mask=mask, training=training,
                                        initial_state=initial_state)


if __name__ == '__main__':
    print('Reloaded')
