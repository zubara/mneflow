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
#from tensorflow.keras.activations import relu
from tensorflow.keras import constraints as k_con, regularizers as k_reg, saving


from tensorflow.keras.layers import Dense, Activation, Conv2D
from tensorflow.keras.layers import Concatenate, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np

bias_const = 0.1
bias_traiable = True

class BaseLayer(tf.keras.layers.Layer):
    def __init__(self, size, nonlin, specs, **args):
        super(BaseLayer, self).__init__(**args)
        self.size = size
        self.nonlin = nonlin
        self.specs = specs
        # self.specs.setdefault('l1_lambda', 0.)
        # self.specs.setdefault('l2_lambda', 0.)
        # self.specs.setdefault('l1_scope', [])
        # self.specs.setdefault('l2_scope', [])
        # self.specs.setdefault('maxnorm_scope', [])

    def _set_regularizer(self):
        if self.scope in self.specs['l1_scope'] or 'weights' in self.specs['l1_scope']:
            reg = k_reg.l1(self.specs['l1_lambda'])
            print('Setting reg for {}, to l1'.format(self.scope))
        elif self.scope in self.specs['l2_scope'] or 'weights' in self.specs['l2_scope']:
            reg = k_reg.l2(self.specs['l2_lambda'])
            print('Setting reg for {}, to l2'.format(self.scope))
        else:
            reg = None
        return reg

    def _set_constraints(self, axis=0):
        if self.scope in self.specs['unitnorm_scope']:
            constr = k_con.UnitNorm(axis=axis)
            print('Setting constraint for {}, to UnitNorm'.format(self.scope))
        else:
            constr = None
        return constr

@saving.register_keras_serializable(package="mneflow")
class FullyConnected(BaseLayer, tf.keras.layers.Layer):


    """
    Fully-connected layer

    """
    def __init__(self, scope='fc', size=None, nonlin=tf.identity, specs={},
                 **args):
        self.scope = scope
        super(FullyConnected, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)
        self.constraint = self._set_constraints()
        self.reg = self._set_regularizer()

    def get_config(self):
        base_config = super(FullyConnected, self).get_config()
        config = {'scope': self.scope, 'size': self.size,
                  'nonlin': self.nonlin, 'specs': self.specs}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        super(FullyConnected, self).build(input_shape)
        # print(input_shape)
        self.flatsize = np.prod(input_shape[1:])
        #print(self.scope, ':::', )

        self.w = self.add_weight(shape=[self.flatsize, self.size],
                                 initializer='he_uniform',
                                 regularizer=self.reg,
                                 constraint=self.constraint,
                                 trainable=True,
                                 name='fc_weights',
                                 dtype=tf.float32)

        self.b = self.add_weight(shape=[self.size],
                                 initializer=Constant(bias_const),
                                 regularizer=None,
                                 trainable=bias_traiable,
                                 name='fc_bias',
                                 dtype=tf.float32)

        print("Built: {} input: {}".format(self.scope, input_shape))


    def call(self, x, training=None):
        """
        FullyConnected layer currying, to apply layer to any input tensor `x`
        """
        while True:
            with tf.name_scope(self.scope):
                if len(x.shape) > 2:  # flatten if input is not 2d array
                    x = tf.reshape(x, [-1, self.flatsize])
                tmp = tf.matmul(x, self.w) + self.b
                tmp = self.nonlin(tmp) #, name='out'
                #print(self.scope, ": output :", tmp.shape)
                return tmp

@saving.register_keras_serializable(package="mneflow")
class DeMixing(BaseLayer):
    """
    Spatial demixing Layer

    """

    def __init__(self, scope="dmx", size=None, nonlin=tf.identity, axis=-1,
                 specs={},  **args):
        self.scope = scope
        self.axis = axis
        super(DeMixing, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)

    def get_config(self):
        config = super(DeMixing, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                        'nonlin': self.nonlin, 'axis': self.axis,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):

        super(DeMixing, self).build(input_shape)
        self.constraint = self._set_constraints(axis=0)
        self.reg = self._set_regularizer()

        self.w = self.add_weight(
                shape=(input_shape[self.axis], self.size),
                initializer='he_uniform',
                regularizer=self.reg,
                constraint = self.constraint,
                trainable=True,
                name='dmx_weights',
                dtype=tf.float32)

        self.b_in = self.add_weight(shape=([self.size]),
                                    initializer=Constant(bias_const),
                                    regularizer=None,
                                    trainable=bias_traiable,
                                    name='bias',
                                    dtype=tf.float32)
        print("Built: {} input: {}".format(self.scope, input_shape))

    #@tf.function
    def call(self, x, training=None):
        """
        """
        while True:
            with tf.name_scope(self.scope):
                try:
                    demix = tf.tensordot(x, self.w, axes=[[self.axis], [0]],
                                         name='dmx')
                    demix = self.nonlin(demix + self.b_in)
                    #print(self.scope, ": output :", demix.shape)
                    return demix
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)
@saving.register_keras_serializable(package="mneflow")
class SquareSymm(BaseLayer):
    """
    SquaredSymmetric Layer

    """
    def __init__(self, scope='ssym', size=None, nonlin=tf.identity, axis=1,
                 specs={},  **args):
        self.scope = scope
        self.axis = axis
        super(SquareSymm, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)

    def get_config(self):
        config = super(SquareSymm, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                        'nonlin': self.nonlin, 'axis': self.axis,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):

        super(SquareSymm, self).build(input_shape)
        self.constraint = self._set_constraints(axis=0)
        self.reg = self._set_regularizer()

        self.w = self.add_weight(
                shape=(input_shape[self.axis], self.size),
                initializer='he_uniform',
                regularizer=self.reg,
                constraint = self.constraint,
                trainable=True,
                name='ssym_weights',
                dtype=tf.float32)

        self.b_in = self.add_weight(shape=([self.size]),
                                    initializer=Constant(0.1),
                                    regularizer=None,
                                    trainable=True,
                                    name='bias',
                                    dtype=tf.float32)
        print("Built: {} input: {}".format(self.scope, input_shape))

    #@tf.function
    def call(self, x, training=None):
        """
        """
        while True:
            with tf.name_scope(self.scope):
                try:
                    d1 = tf.tensordot(x, self.w, axes=[[1], [0]],
                                         name='smx') #output
                    d2 = tf.tensordot(d1, self.w, axes=[[1], [0]],
                                         name='smx')

                    demix = self.nonlin(d2 + self.b_in)
                    return demix
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)

@saving.register_keras_serializable(package="mneflow")
class LFTConv(BaseLayer):
    """
    Stackable temporal convolutional layer, interpreatble (LF)

    """

    def __init__(self, scope='tconv', size=32,  nonlin=tf.nn.relu,
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
                        'nonlin': self.nonlin, 'padding': self.padding,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        super(LFTConv, self).build(input_shape)
        self.constraint = self._set_constraints(axis=1)
        self.reg = self._set_regularizer()
        shape = [1, self.filter_length, input_shape[-1], 1]
        self.filters = self.add_weight(shape=shape,
                                       initializer='he_uniform',
                                       regularizer=self.reg,
                                       constraint=self.constraint,
                                       trainable=True,
                                       name='tconv_weights',
                                       dtype=tf.float32)

        self.b = self.add_weight(shape=([input_shape[-1]]),
                                 initializer=Constant(bias_const),
                                 regularizer=None,
                                 trainable=bias_traiable,
                                 name='bias',
                                 dtype=tf.float32)
        print("Built: {} input: {}".format(self.scope, input_shape))

    #@tf.function
    def call(self, x, training=None):
        """
        """
        while True:
            with tf.name_scope(self.scope):
                try:
                    conv = tf.nn.depthwise_conv2d(x,
                                                  self.filters,
                                                  padding=self.padding,
                                                  strides=[1, 1, 1, 1],
                                                  data_format='NHWC')
                    conv = self.nonlin(conv + self.b)

                    #print(self.scope, ": output :", conv.shape)
                    return conv
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)

@saving.register_keras_serializable(package="mneflow")
class LFTConvTranspose(BaseLayer):
    """
    DepthWise temporal deconvolutional layer, interpreatble

    """

    def __init__(self, target_shape,
                 # scope='deconv',
                 # nonlin=tf.nn.identity,
                 filter_length=7,
                 padding='SAME',
                 stride=2,
                 #specs={},
                 **args):

        self.scope = 'deconv'
        self.padding = padding

        self.target_shape = target_shape
        print(target_shape, self.target_shape)
        #print(self.target_shape)
        self.kernel_shape = [1, filter_length]
        self.strides = [1, stride]
        self.stride = stride
        super(LFTConvTranspose, self).__init__(size=48,
                                               #scope=self.scope,
                                               nonlin=tf.identity,
                                               specs={}
                                               )

        #self.size = size
        #self.filter_length = filter_length


    def get_config(self):

        config = super(LFTConvTranspose, self).get_config()
        config.update({'scope': self.scope,
                        'filter_length': self.filter_length,
                        'nonlin': self.nonlin, 'padding': self.padding,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):

        #self.constraint = self._set_constraints(axis=1)
        #self.reg = self._set_regularizer()

        self.input_shape = input_shape
        print(self.target_shape)
        self.n_channels = input_shape[-1]
        default_output_t = input_shape[2] * self.stride
        diff_padding = default_output_t - self.target_shape[2]
        self.n_pads = max(0, min(self.stride-1, diff_padding))
        # Each channel gets its own transpose convolution
        self.deconv_units = [tf.keras.layers.Conv2DTranspose(filters=1,  # Each channel processed independently
                            kernel_size=self.kernel_shape,
                            strides=self.stride,
                            padding=[[0, 0], [0, 0],
                                     [self.n_pads, 0], [0, 0]]) for i in range(self.n_channels)]

        super(LFTConvTranspose).__init__()
        # self.filters = [self.add_weight(shape=self.kernel_shape,
        #                                initializer='he_uniform',
        #                                #regularizer=self.reg,
        #                                #constraint=self.constraint,
        #                                trainable=True,
        #                                name='deconv_weights',
        #                                dtype=tf.float32)
        #                 for i in range(self.n_channels)]


        # self.b = [self.add_weight(shape=([1]),
        #                          initializer=Constant(bias_const),
        #                          regularizer=None,
        #                          trainable=bias_traiable,
        #                          name='bias',
        #                          dtype=tf.float32)
        #           for i in range(self.n_channels)]

        print("Built: {} input: {}".format(self.scope, input_shape))

    #@tf.function
    def call(self, x, training=None):
        """
        """
        while True:
             with tf.name_scope(self.scope):
                 try:
                    split_tensors = tf.split(x, self.n_channels, axis=-1)
                    print("split:", split_tensors[0].shape)
                    # Process each channel separately with its own Conv2DTranspose
                    transposed_tensors = [self.deconv_units[i](split_tensors[i]) for i in range(self.n_channels)]

                    # Concatenate all channels back together
                    concatenated = tf.concat(transposed_tensors, axis=-1)
                    print("concatenated:", concatenated.shape)
                 except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)
        # while True:
        #     with tf.name_scope(self.scope):
        #         try:
        #             # Split the input tensor into individual channels
        #             split_tensors = tf.split(x, self.n_channels, axis=-1)
        #             print("split:", split_tensors[0].shape)
        #             transposed_tensors = []
        #             for i in range(self.n_channels):
        #                 # Each channel gets its own transpose convolution
        #                 transposed = tf.nn.conv2d_transpose(split_tensors[i],
        #                                 filters=self.filters[i],
        #                                 output_shape=[self.input_shape[0], self.target_shape[1:3], 1],
        #                                 strides=self.strides,
        #                                 padding=[[0, 0], [0, 0],
        #                                          [self.n_pads, 0], [0, 0]]) + self.b[i]
        #                 if i == 0:
        #                     print("1 transposed:", transposed.shape)
        #                 transposed_tensors.append(transposed)

        #             # Concatenate all channels back together
        #             concatenated = tf.concat(transposed_tensors, axis=-1)
        #             print("concatenated:", concatenated.shape)
        #             # conv = tf.nn.depthwise_conv2d(x,
        #             #                               self.filters,
        #             #                               padding=self.padding,
        #             #                               strides=[1, 1, 1, 1],
        #             #                               data_format='NHWC')
        #             # conv = self.nonlin(conv + self.b)

        #             #print(self.scope, ": output :", conv.shape)
        #             #assert concatenated.shape == self.target_shape
        #             return concatenated
        #         except(AttributeError):
        #             input_shape = x.shape
        #             self.build(input_shape)

@saving.register_keras_serializable(package="mneflow")
class VARConv(BaseLayer):
    """
    Stackable temporal convolutional layer
    """

    def __init__(self, scope='tconv', size=32,  nonlin=tf.nn.relu,
                 filter_length=7, pooling=2, padding='SAME', specs={},
                 **args):
        self.scope = scope
        super(VARConv, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)
        self.size = size
        self.nonlin = nonlin
        self.filter_length = filter_length
        self.padding = padding


    def get_config(self):

        config = super(VARConv, self).get_config()
        config.update({'scope': self.scope,
                        'filter_length': self.filter_length,
                        'nonlin': self.nonlin, 'padding': self.padding,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        print("input_shape:", input_shape)
        super(VARConv, self).build(input_shape)

        self.constraint = self._set_constraints()
        self.reg = self._set_regularizer()
        shape = [1, self.filter_length, input_shape[-1], self.size]
        self.filters = self.add_weight(shape=shape,
                                       initializer='he_uniform',
                                       regularizer=self.reg,
                                       constraint=self.constraint,
                                       trainable=True,
                                       name='tconv_weights',
                                       dtype=tf.float32)

        self.b = self.add_weight(shape=([input_shape[-1]]),
                                 initializer=Constant(bias_const),
                                 regularizer=None,
                                 trainable=bias_traiable,
                                 name='bias',
                                 dtype=tf.float32)
        print("Built: {} input: {}".format(self.scope, input_shape))

    #@tf.function
    def call(self, x, training=None):
        """
        """
        while True:
            with tf.name_scope(self.scope):
                try:
                    conv = tf.nn.conv2d(x, self.filters,
                                        padding=self.padding,
                                        strides=[1, 1, 1, 1],
                                        data_format='NHWC')

                    conv = self.nonlin(conv + self.b)
                    #print(self.scope, ": output :", conv.shape)
                    return conv
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)
                    #print(self.scope, 'building from call')

@saving.register_keras_serializable(package="mneflow")
class TempPooling(BaseLayer):
    def __init__(self, scope='pool', stride=2, pooling=2, specs={},
                 padding='SAME', pool_type='max', **args):
        self.scope = '_'.join([pool_type, scope])
        super(TempPooling, self).__init__(size=None, nonlin=None, specs=specs,
                                          **args)
        self.strides = [1, 1, stride,  1]
        self.kernel = [1, 1, pooling,  1]

        self.padding = padding
        self.pool_type = pool_type

    def get_config(self):

        config = super(TempPooling, self).get_config()
        config.update({'scope': self.scope,
                        'stride': self.strides[2],
                        'pooling': self.kernel[2], 'padding': self.padding,
                        'specs':self.specs, 'pool_type' : self.pool_type})
        return config


    #@tf.function
    def call(self, x):
        """
        """
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
        #print(self.scope, ": output :", pooled.shape)
        return pooled

    def build(self, input_shape):
        super(TempPooling, self).build(input_shape)
        self.built = True


    # def get_config(self):
    #     config = super(TempPooling, self).get_config()
    #     config.update({'scope': self.scope,
    #                    'pool_type': self.pool_type,
    #                    'stride': self.strides, 'pooling': self.pooling,
    #                    'padding': self.padding})
    #     return config


@saving.register_keras_serializable(package="mneflow")
class LSTM(tf.keras.layers.LSTM):
    def __init__(self, scope='lstm', size=32, nonlin='tanh', dropout=0.0,
                 recurrent_activation='tanh', recurrent_dropout=0.0,
                 use_bias=True, unit_forget_bias=True,
                 kernel_regularizer=None, bias_regularizer=None,
                 return_sequences=True, stateful=False, unroll=False, **args):
        super(LSTM, self).__init__(name=scope,
                                     units=size,
                                     activation=nonlin,
                                     dropout=dropout,
                                     recurrent_activation=recurrent_activation,
                                     recurrent_dropout=recurrent_dropout,
                                     use_bias=use_bias,
                                     unit_forget_bias=unit_forget_bias,
                                     kernel_regularizer=kernel_regularizer,
                                     # kernel_initializer='glorot_uniform',
                                     # recurrent_initializer='orthogonal',
                                     bias_regularizer=bias_regularizer,
                                     return_sequences=return_sequences,
                                     stateful=stateful,
                                     unroll=unroll,
                                     **args)
        self.scope = scope
        self.size = size
        self.nonlin = nonlin
        print(self.scope, 'init : OK')

    def get_config(self):
        config = super(LSTM, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                        'nonlin': self.nonlin})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        # print(self.scope, 'build : OK')
        super(LSTM, self).build(input_shape)

    @tf.function
    def call(self, inputs, mask=None, training=None, initial_state=None):
        # print(self.scope, inputs.shape)
        return super(LSTM, self).call(inputs, mask=mask, training=training,
                                        initial_state=initial_state)


@saving.register_keras_serializable(package="mneflow")
class WeightedSum(BaseLayer):
    """Compute weighted sum over rows using same weights for each input
    channel.
    """

    def __init__(self, scope='wsum', size=None, nonlin=tf.identity, axis=1,
                 specs=None, **args):
        if specs is None:
            specs = dict()
        self.scope = scope
        self.axis = axis
        super().__init__(size=size,
                         nonlin=nonlin,
                         specs=specs,
                         **args)

    def get_config(self):
        config = super().get_config()
        config.update({'scope': self.scope,
                       'size': self.size,
                       'nonlin': self.nonlin,
                       'axis': self.axis,
                       'specs': self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        cls.scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        super().build(input_shape)
        self.constraint = self._set_constraints(axis=0)
        self.reg = self._set_regularizer()

        self.w = self.add_weight(
            shape=(input_shape[self.axis], self.size),
            initializer='he_uniform',
            regularizer=self.reg,
            constraint=self.constraint,
            trainable=True,
            name='wsum_weights',
            dtype=tf.float32)

        self.b_in = self.add_weight(shape=([self.size]),
                                    initializer=Constant(0.1),
                                    regularizer=None,
                                    trainable=True,
                                    name='bias',
                                    dtype=tf.float32)
        print(f"Built: {self.scope} input: {input_shape}")

    # @tf.function
    def call(self, x, training=None):
        while True:
            with tf.name_scope(self.scope):
                try:
                    d1 = tf.tensordot(x,
                                      self.w,
                                      axes=[[1], [0]],
                                      name='smx')  # output
                    if self.nonlin:
                        demix = self.nonlin(d1 + self.b_in)
                    else:
                        demix = d1 + self.b_in
                    return demix
                except AttributeError:
                    input_shape = x.shape
                    self.build(input_shape)


@saving.register_keras_serializable(package="mneflow")
class WeightSum3d(BaseLayer):
    """Compute weighted sum over rows using different weights for each input
    channel (frequency): depthwise convolution.
    """

    def __init__(self, scope='wsum3d', size=None, nonlin=tf.identity, axis=1,
                 specs={}, **args):
        self.scope = scope
        self.axis = axis
        super().__init__(
            size=size,
            nonlin=nonlin,
            specs=specs,
            **args)

    def get_config(self):
        config = super(WeightSum3d, self).get_config()
        config.update({'scope': self.scope,
                       'size': self.size,
                       'nonlin': self.nonlin,
                       'axis': self.axis,
                       'specs': self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        cls.scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        super().build(input_shape)
        self.constraint = self._set_constraints(axis=0)
        self.reg = self._set_regularizer()

        self.w = self.add_weight(
            shape=(input_shape[-1], input_shape[self.axis], self.size),
            initializer='he_uniform',
            regularizer=self.reg,
            constraint=self.constraint,
            trainable=True,
            name='wsum_weights',
            dtype=tf.float32)

        self.b_in = self.add_weight(
            shape=(input_shape[-1], self.size),
            initializer=Constant(0.1),
            regularizer=None,
            trainable=True,
            name='bias',
            dtype=tf.float32)
        print(f"Built: {self.scope} input: {input_shape}")

    # @tf.function
    def call(self, x, training=None):
        while True:
            with tf.name_scope(self.scope):
                try:
                    out = []
                    for channel in range(x.shape[-1]):
                        row_sum = tf.tensordot(
                            x[..., channel],
                            self.w[channel],
                            axes=[[1], [0]],
                            name='smx')  # output
                        demix = self.nonlin(row_sum + self.b_in[channel])
                        out.append(demix)
                    result = tf.stack(out, axis=-1)
                    return result
                except AttributeError:
                    input_shape = x.shape
                    self.build(input_shape)


@saving.register_keras_serializable(package="mneflow")
class SquareSum3d(BaseLayer):
    """Compute weighted sum over rows and columns using the same weights for
    both, but different weights for each input channel (frequency): depthwise
    convolution.
    """

    def __init__(self, scope='wsum3d', size=None, nonlin=tf.identity, axis=1,
                 specs={}, **args):
        self.scope = scope
        self.axis = axis
        super(SquareSum3d, self).__init__(size=size, nonlin=nonlin, specs=specs,
                                          **args)

    def get_config(self):
        config = super(SquareSum3d, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'nonlin': self.nonlin, 'axis': self.axis,
                       'specs': self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):

        super(SquareSum3d, self).build(input_shape)
        self.constraint = self._set_constraints(axis=0)
        self.reg = self._set_regularizer()

        self.w = self.add_weight(
            shape=(input_shape[-1], input_shape[self.axis], self.size),
            initializer='he_uniform',
            regularizer=self.reg,
            constraint=self.constraint,
            trainable=True,
            name='wsum_weights',
            dtype=tf.float32)

        self.b_in = self.add_weight(shape=(input_shape[-1], self.size),
                                    initializer=Constant(0.1),
                                    regularizer=None,
                                    trainable=True,
                                    name='bias',
                                    dtype=tf.float32)
        print("Built: {} input: {}".format(self.scope, input_shape))

    # @tf.function
    def call(self, x, training=None):
        while True:
            with tf.name_scope(self.scope):
                try:
                    out = []
                    for channel in range(x.shape[-1]):
                        row_sum = tf.tensordot(x[..., channel],
                                               self.w[channel],
                                               axes=[[1], [0]],
                                               name='smx')  # output
                        col_sum = tf.tensordot(row_sum,
                                               self.w[channel],
                                               axes=[[1], [0]],
                                               name='smx')
                        demix = self.nonlin(col_sum + self.b_in[channel])
                        out.append(demix)
                    result = tf.stack(out, axis=-1)
                    return result
                except AttributeError:
                    input_shape = x.shape
                    self.build(input_shape)


def soft_attention(x):

    attention_weights = Dense(units=1, activation="sigmoid")(x)
    #print(attention_weights.shape)
    #attention_weights = tf.keras.layers.Reshape((-1,))(attention_weights)
   # print(attention_weights.shape)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)
    #print(attention_weights.shape)
    attention_output = tf.keras.layers.Dot(axes=[1,1])([attention_weights, x])

    return attention_output


def se_block(x, ratio=4):
    """
    Squeeze and excitation (SE) channel attention block.

    Parameters
    ----------
    x : array_like
        Input data array for the channel attention
    ratio : integer, optional
        Reduction parameter to reduce the number of channels.
        The default is 4.

    Returns
    -------
    out : tensor
        The input array multiplied by the channel attention scores.
    """
    b, _, _, c = x.shape
    y = GlobalAveragePooling2D()(x)

    y = Dense(c // ratio, activation='relu')(y)
    y = Dense(c, activation='sigmoid')(y)
    out = Multiply()([x, y])
    return out


def channel_attention(x, ratio=4):
    """
    Channel attention layer for the CBAM attention block.

    Parameters
    ----------
    x : tensor
        Input data array for the channel attention.
    ratio : integer, optional
        Reduction parameter to reduce the number of channels.
        The default is 4.

    Returns
    -------
    out : tensor
        The input array multiplied by the channel attention scores.

    """
    b, _, _, c = x.shape

    l1 = Dense(units=c // ratio, activation=tf.nn.relu, use_bias=False)
    l2 = Dense(units=c, use_bias=False)

    x1 = GlobalAveragePooling2D()(x)
    x1 = l1(x1)
    x1 = l2(x1)

    x2 = GlobalMaxPooling2D()(x)
    x2 = l1(x2)
    x2 = l2(x2)

    # Sum and apply sigmoid
    features = x1 + x2
    features = Activation("sigmoid")(features)

    out = Multiply()([x, features])
    return out


def spatial_attention(x):
    """
    Spatial attention layer for the CBAM attention block.

    Parameters
    ----------
    x : tensor
        Input data array for the spatial attention.

    Returns
    -------
    features : tensor
        The input array multiplied by the spatial attention socres.

    """
    # Average pooling
    x1 = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)

    # Max pooling
    x2 = tf.keras.layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)

    # concat
    features = Concatenate()([x1, x2])

    # conv
    features = Conv2D(1, kernel_size=(7, 7), padding="same",
                      activation="sigmoid")(features)

    features = Multiply()([x, features])

    return features


def cbam(x, ratio=4):
    """CBAM attention block using channel attention and spatial attention.

    Parameters
    ----------
    x : array_like
        Input data array for the channel attention.
    ratio : integer, optional
        Reduction parameter to reduce the number of channels.
        The default is 4.

    Returns
    -------
    x : tensor
        The input array multiplied by the spatial and channel attention scores.
    """
    x = channel_attention(x, ratio)
    x = spatial_attention(x)
    return x


if __name__ == '__main__':
    print('Reloaded')

