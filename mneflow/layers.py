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
    """Common base class for mneflow's custom Keras layers.

    Stores the layer's output ``size``, nonlinearity, and ``specs``
    dict, and provides helpers to derive a weight regularizer and
    constraint from ``specs`` based on the subclass's ``self.scope``.

    Parameters
    ----------
    size : int or None
        Output size of the layer (e.g. number of units/components).

    nonlin : callable
        Nonlinearity (activation function) applied to the layer's
        output.

    specs : dict
        Regularization/constraint specification dict. Expected to
        contain (depending on which helper is used) ``'l1_scope'``,
        ``'l1_lambda'``, ``'l2_scope'``, ``'l2_lambda'``, and
        ``'unitnorm_scope'``.

    **args : dict
        Additional keyword arguments passed to
        ``tf.keras.layers.Layer.__init__``.

    """
    def __init__(self, size, nonlin, specs, **args):
        """Store the layer's size, nonlinearity, and specs.

        Parameters
        ----------
        size : int or None
            Output size of the layer.

        nonlin : callable
            Nonlinearity applied to the layer's output.

        specs : dict
            Regularization/constraint specification dict.

        **args : dict
            Additional keyword arguments passed to
            ``tf.keras.layers.Layer.__init__``.

        """
        super(BaseLayer, self).__init__(**args)
        self.size = size
        self.nonlin = nonlin
        self.specs = specs

    def _set_regularizer(self):
        """Build a weight regularizer for this layer based on ``self.specs``.

        Returns
        -------
        reg : tf.keras.regularizers.Regularizer or None
            An L1 regularizer if ``self.scope`` (or ``'weights'``) is
            in ``self.specs['l1_scope']``, an L2 regularizer if it is
            in ``self.specs['l2_scope']``, otherwise None.

        """
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
        """Build a weight constraint for this layer based on ``self.specs``.

        Parameters
        ----------
        axis : int, optional
            Axis along which to apply the unit-norm constraint, if
            applicable. Defaults to 0.

        Returns
        -------
        constr : tf.keras.constraints.Constraint or None
            A ``UnitNorm(axis=axis)`` constraint if ``self.scope`` is
            in ``self.specs['unitnorm_scope']``, otherwise None.

        """
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
        """Initialize a fully-connected layer.

        Parameters
        ----------
        scope : str, optional
            Name scope / regularization-and-constraint lookup key for
            this layer. Defaults to 'fc'.

        size : int, optional
            Number of output units. Defaults to None.

        nonlin : callable, optional
            Nonlinearity applied to the layer's output. Defaults to
            ``tf.identity``.

        specs : dict, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Defaults to ``{}``.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        self.scope = scope
        super(FullyConnected, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)
        self.constraint = self._set_constraints()
        self.reg = self._set_regularizer()

    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base ``tf.keras.layers.Layer`` config merged with
            ``'scope'``, ``'size'``, ``'nonlin'``, and ``'specs'``.

        """
        base_config = super(FullyConnected, self).get_config()
        config = {'scope': self.scope, 'size': self.size,
                  'nonlin': self.nonlin, 'specs': self.specs}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : FullyConnected
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create the layer's weights.

        Flattens all but the batch dimension of ``input_shape`` and
        creates a dense weight matrix ``'fc_weights'`` of shape
        ``[flatsize, size]`` and bias ``'fc_bias'`` of shape
        ``[size]``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """
        super(FullyConnected, self).build(input_shape)
        self.flatsize = np.prod(input_shape[1:])

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

        Parameters
        ----------
        x : tf.Tensor
            Input tensor. Flattened (except for the batch dimension)
            before the dense transform if it has more than 2 axes.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        tmp : tf.Tensor, shape (batch, size)
            ``nonlin(x @ w + b)``.

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
        """Initialize a spatial demixing layer.

        Parameters
        ----------
        scope : str, optional
            Name scope / regularization-and-constraint lookup key for
            this layer. Defaults to 'dmx'.

        size : int, optional
            Number of output (latent) components. Defaults to None.

        nonlin : callable, optional
            Nonlinearity applied to the layer's output. Defaults to
            ``tf.identity``.

        axis : int, optional
            Axis of the input tensor to contract the demixing weights
            against. Defaults to -1.

        specs : dict, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Defaults to ``{}``.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        self.scope = scope
        self.axis = axis
        super(DeMixing, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)

    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'size'``,
            ``'nonlin'``, ``'axis'``, and ``'specs'``.

        """
        config = super(DeMixing, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                        'nonlin': self.nonlin, 'axis': self.axis,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : DeMixing
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create the layer's demixing weights and bias.

        Creates weight matrix ``'dmx_weights'`` of shape
        ``[input_shape[axis], size]`` and bias ``'bias'`` of shape
        ``[size]``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """

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
        """Apply the spatial demixing transform to the input tensor.

        Contracts ``x`` with the demixing weights along ``self.axis``
        and applies the nonlinearity. Builds the layer on first call
        if it has not been built yet.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        demix : tf.Tensor
            ``nonlin(tensordot(x, w, axes=[[axis], [0]]) + b_in)``.

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
    Squared-symmetric (congruence transform) layer.

    Applies the same weight matrix along two dimensions of a
    per-sample square matrix, i.e. computes ``w.T @ X @ w`` for each
    sample ``X`` (see :meth:`call`). Commonly used to project a square
    spatial covariance/connectivity matrix of shape
    ``(n_channels, n_channels)`` onto a smaller, still-square
    ``(size, size)`` matrix while preserving symmetry.

    Shape contract
    --------------
    Expects a 3-D input of shape ``(batch, N, N)`` where the two
    non-batch dimensions are equal, i.e.
    ``input_shape[1] == input_shape[-1]``. This is a constraint on the
    *input* only -- ``size`` is independent of ``N`` and may be
    smaller, equal to, or larger than it. Passing a non-square input
    raises an error from the second ``tf.tensordot`` call in
    :meth:`call`, since its two contracted axes would then have
    mismatched sizes.

    Output shape: ``(batch, size, size)``.

    """
    def __init__(self, scope='ssym', size=None, nonlin=tf.identity, axis=1,
                 specs={},  **args):
        """Initialize a squared-symmetric layer.

        Parameters
        ----------
        scope : str, optional
            Name scope / regularization-and-constraint lookup key for
            this layer. Defaults to 'ssym'.

        size : int, optional
            Number of output components. Defaults to None.

        nonlin : callable, optional
            Nonlinearity applied to the layer's output. Defaults to
            ``tf.identity``.

        axis : int, optional
            Axis of the input tensor to contract the weights against
            (applied twice, see :meth:`call`). Defaults to 1.

        specs : dict, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Defaults to ``{}``.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        self.scope = scope
        self.axis = axis
        super(SquareSymm, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)

    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'size'``,
            ``'nonlin'``, ``'axis'``, and ``'specs'``.

        """
        config = super(SquareSymm, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                        'nonlin': self.nonlin, 'axis': self.axis,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : SquareSymm
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create the layer's weights and bias.

        Creates weight matrix ``'ssym_weights'`` of shape
        ``[input_shape[axis], size]`` (applied twice, see
        :meth:`call`) and bias ``'bias'`` of shape ``[size]``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """

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
        """Apply the squared-symmetric transform to the input tensor.

        Contracts ``x`` with the layer's weight matrix along axis 1
        twice in succession (``d1 = x @ w``, ``d2 = d1 @ w``), then
        applies the nonlinearity. Builds the layer on first call if
        it has not been built yet.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        demix : tf.Tensor
            ``nonlin(d2 + b_in)``, where ``d2`` is ``w`` applied twice
            to ``x``.

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
        """Initialize a depthwise temporal convolutional layer.

        Parameters
        ----------
        scope : str, optional
            Name scope / regularization-and-constraint lookup key for
            this layer. Defaults to 'tconv'.

        size : int, optional
            Currently unused by this layer's own weights (kept for
            interface consistency with :class:`VARConv`); the number
            of output channels is determined by the input's last
            dimension. Defaults to 32.

        nonlin : callable, optional
            Nonlinearity applied to the layer's output. Defaults to
            ``tf.nn.relu``.

        filter_length : int, optional
            Length (in time points) of the depthwise convolution
            kernel. Defaults to 7.

        pooling : int, optional
            Currently unused by this layer. Defaults to 2.

        padding : str, optional
            Padding mode passed to
            ``tf.nn.depthwise_conv2d``. Defaults to 'SAME'.

        specs : dict, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Defaults to ``{}``.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        self.scope = scope
        super(LFTConv, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)
        self.size = size
        self.filter_length = filter_length
        self.padding = padding

    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'filter_length'``,
            ``'nonlin'``, ``'padding'``, and ``'specs'``.

        """

        config = super(LFTConv, self).get_config()
        config.update({'scope': self.scope,
                        'filter_length': self.filter_length,
                        'nonlin': self.nonlin, 'padding': self.padding,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : LFTConv
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create the layer's depthwise convolution filters and bias.

        Creates filters ``'tconv_weights'`` of shape
        ``[1, filter_length, input_shape[-1], 1]`` and bias
        ``'bias'`` of shape ``[input_shape[-1]]``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """
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
        """Apply the depthwise temporal convolution to the input tensor.

        Builds the layer on first call if it has not been built yet.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        conv : tf.Tensor
            ``nonlin(depthwise_conv2d(x, filters) + b)``.

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
        """Initialize a depthwise temporal transposed-convolution layer.

        Parameters
        ----------
        target_shape : tuple of int
            Desired output shape (used to compute end-padding so
            that the deconvolved time axis matches
            ``target_shape[2]``).

        filter_length : int, optional
            Length (in time points) of each per-channel transposed
            convolution kernel. Defaults to 7.

        padding : str, optional
            Currently unused (padding is instead computed explicitly
            from ``target_shape`` in :meth:`build`). Defaults to
            'SAME'.

        stride : int, optional
            Temporal stride of the transposed convolution. Defaults
            to 2.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """

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
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'filter_length'``,
            ``'nonlin'``, ``'padding'``, and ``'specs'``.

        """

        config = super(LFTConvTranspose, self).get_config()
        config.update({'scope': self.scope,
                        'filter_length': self.filter_length,
                        'nonlin': self.nonlin, 'padding': self.padding,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : LFTConvTranspose
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create one ``Conv2DTranspose`` sub-layer per input channel.

        Each channel of the input is deconvolved independently (via
        ``self.deconv_units``), with left-padding chosen so that the
        deconvolved time axis matches ``self.target_shape[2]``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """

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
 

        print("Built: {} input: {}".format(self.scope, input_shape))

    #@tf.function
    def call(self, x, training=None):
        """Apply the per-channel transposed convolution to the input tensor.

        Splits ``x`` into per-channel tensors, applies each channel's
        ``Conv2DTranspose`` sub-layer, and concatenates the results
        back along the channel axis. Builds the layer on first call
        if it has not been built yet.

        Note that this method does not currently return the
        concatenated result (see the commented-out alternative
        implementation below it).

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        None

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
 

@saving.register_keras_serializable(package="mneflow")
class VARConv(BaseLayer):
    """
    Stackable temporal convolutional layer
    """

    def __init__(self, scope='tconv', size=32,  nonlin=tf.nn.relu,
                 filter_length=7, pooling=2, padding='SAME', specs={},
                 **args):
        """Initialize a (non-depthwise) temporal convolutional layer.

        Parameters
        ----------
        scope : str, optional
            Name scope / regularization-and-constraint lookup key for
            this layer. Defaults to 'tconv'.

        size : int, optional
            Number of output convolutional filters. Defaults to 32.

        nonlin : callable, optional
            Nonlinearity applied to the layer's output. Defaults to
            ``tf.nn.relu``.

        filter_length : int, optional
            Length (in time points) of the convolution kernel.
            Defaults to 7.

        pooling : int, optional
            Currently unused by this layer. Defaults to 2.

        padding : str, optional
            Padding mode passed to ``tf.nn.conv2d``. Defaults to
            'SAME'.

        specs : dict, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Defaults to ``{}``.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        self.scope = scope
        super(VARConv, self).__init__(size=size, nonlin=nonlin, specs=specs,
             **args)
        self.size = size
        self.nonlin = nonlin
        self.filter_length = filter_length
        self.padding = padding


    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'filter_length'``,
            ``'nonlin'``, ``'padding'``, and ``'specs'``.

        """

        config = super(VARConv, self).get_config()
        config.update({'scope': self.scope,
                        'filter_length': self.filter_length,
                        'nonlin': self.nonlin, 'padding': self.padding,
                        'specs':self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : VARConv
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create the layer's convolution filters and bias.

        Creates filters ``'tconv_weights'`` of shape
        ``[1, filter_length, input_shape[-1], size]`` and bias
        ``'bias'`` of shape ``[input_shape[-1]]``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """
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
        """Apply the temporal convolution to the input tensor.

        Builds the layer on first call if it has not been built yet.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        conv : tf.Tensor
            ``nonlin(conv2d(x, filters) + b)``.

        """
        while True:
            with tf.name_scope(self.scope):
                try:
                    conv = tf.nn.conv2d(x, self.filters,
                                        padding=self.padding,
                                        strides=[1, 1, 1, 1],
                                        data_format='NHWC')

                    conv = self.nonlin(conv + self.b)
                    return conv
                except(AttributeError):
                    input_shape = x.shape
                    self.build(input_shape)

@saving.register_keras_serializable(package="mneflow")
class TempPooling(BaseLayer):
    """Temporal (max or average) pooling layer.

    """
    def __init__(self, scope='pool', stride=2, pooling=2, specs={},
                 padding='SAME', pool_type='max', **args):
        """Initialize a temporal pooling layer.

        Parameters
        ----------
        scope : str, optional
            Base name for this layer's scope; the actual scope used
            is ``'<pool_type>_<scope>'``. Defaults to 'pool'.

        stride : int, optional
            Pooling stride along the time axis. Defaults to 2.

        pooling : int, optional
            Pooling window size along the time axis. Defaults to 2.

        specs : dict, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Not used by this layer (it has no
            trainable weights). Defaults to ``{}``.

        padding : str, optional
            Padding mode passed to the pooling op. Defaults to
            'SAME'.

        pool_type : str {'max', 'avg'}, optional
            Type of pooling to apply. Defaults to 'max'.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        self.scope = '_'.join([pool_type, scope])
        super(TempPooling, self).__init__(size=None, nonlin=None, specs=specs,
                                          **args)
        self.strides = [1, 1, stride,  1]
        self.kernel = [1, 1, pooling,  1]

        self.padding = padding
        self.pool_type = pool_type

    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'stride'``,
            ``'pooling'``, ``'padding'``, ``'specs'``, and
            ``'pool_type'``.

        """

        config = super(TempPooling, self).get_config()
        config.update({'scope': self.scope,
                        'stride': self.strides[2],
                        'pooling': self.kernel[2], 'padding': self.padding,
                        'specs':self.specs, 'pool_type' : self.pool_type})
        return config


    #@tf.function
    def call(self, x):
        """Apply temporal max- or average-pooling to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        pooled : tf.Tensor
            Pooled tensor, using ``tf.nn.avg_pool2d`` if
            ``self.pool_type == 'avg'``, otherwise
            ``tf.nn.max_pool2d``.

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
        """Mark the layer as built (no trainable weights to create).

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """
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
    """Thin wrapper around ``tf.keras.layers.LSTM`` using mneflow's
    naming conventions (``scope``, ``size``, ``nonlin``).

    """
    def __init__(self, scope='lstm', size=32, nonlin='tanh', dropout=0.0,
                 recurrent_activation='tanh', recurrent_dropout=0.0,
                 use_bias=True, unit_forget_bias=True,
                 kernel_regularizer=None, bias_regularizer=None,
                 return_sequences=True, stateful=False, unroll=False, **args):
        """Initialize an LSTM layer.

        Parameters
        ----------
        scope : str, optional
            Layer name, passed to ``tf.keras.layers.LSTM`` as
            ``name``. Defaults to 'lstm'.

        size : int, optional
            Number of LSTM units, passed as ``units``. Defaults to
            32.

        nonlin : str or callable, optional
            Activation function, passed as ``activation``. Defaults
            to 'tanh'.

        dropout : float, optional
            Dropout rate for the input transform. Defaults to 0.0.

        recurrent_activation : str or callable, optional
            Activation function for the recurrent step. Defaults to
            'tanh'.

        recurrent_dropout : float, optional
            Dropout rate for the recurrent transform. Defaults to
            0.0.

        use_bias : bool, optional
            Whether the layer uses a bias vector. Defaults to True.

        unit_forget_bias : bool, optional
            Whether to add 1 to the bias of the forget gate at
            initialization. Defaults to True.

        kernel_regularizer : tf.keras.regularizers.Regularizer, optional
            Regularizer for the input kernel weights. Defaults to
            None.

        bias_regularizer : tf.keras.regularizers.Regularizer, optional
            Regularizer for the bias vector. Defaults to None.

        return_sequences : bool, optional
            Whether to return the full output sequence or only the
            last output. Defaults to True.

        stateful : bool, optional
            Whether to reuse the last state for each sample as the
            initial state for the next batch. Defaults to False.

        unroll : bool, optional
            Whether to unroll the network (may be faster for short
            sequences). Defaults to False.

        **args : dict
            Additional keyword arguments passed to
            ``tf.keras.layers.LSTM.__init__``.

        """
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
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base ``tf.keras.layers.LSTM`` config merged with
            ``'scope'``, ``'size'``, and ``'nonlin'``.

        """
        config = super(LSTM, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                        'nonlin': self.nonlin})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : LSTM
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Build the underlying ``tf.keras.layers.LSTM``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """
        # print(self.scope, 'build : OK')
        super(LSTM, self).build(input_shape)

    @tf.function
    def call(self, inputs, mask=None, training=None, initial_state=None):
        """Apply the LSTM to the input sequence.

        Parameters
        ----------
        inputs : tf.Tensor
            Input sequence tensor.

        mask : tf.Tensor, optional
            Boolean mask indicating which timesteps to ignore.
            Defaults to None.

        training : bool, optional
            Whether the layer is in training mode. Defaults to None.

        initial_state : list of tf.Tensor, optional
            Initial hidden/cell state. Defaults to None.

        Returns
        -------
        output : tf.Tensor
            Output of ``tf.keras.layers.LSTM.call``.

        """
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
        """Initialize a weighted-sum layer.

        Parameters
        ----------
        scope : str, optional
            Name scope / regularization-and-constraint lookup key for
            this layer. Defaults to 'wsum'.

        size : int, optional
            Number of output components. Defaults to None.

        nonlin : callable, optional
            Nonlinearity applied to the layer's output. Defaults to
            ``tf.identity``.

        axis : int, optional
            Axis of the input tensor to contract the weights against.
            Defaults to 1.

        specs : dict or None, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Defaults to None, in which case an
            empty dict is used.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        if specs is None:
            specs = dict()
        self.scope = scope
        self.axis = axis
        super().__init__(size=size,
                         nonlin=nonlin,
                         specs=specs,
                         **args)

    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'size'``,
            ``'nonlin'``, ``'axis'``, and ``'specs'``.

        """
        config = super().get_config()
        config.update({'scope': self.scope,
                       'size': self.size,
                       'nonlin': self.nonlin,
                       'axis': self.axis,
                       'specs': self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : WeightedSum
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        cls.scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create the layer's weights and bias.

        Creates weight matrix ``'wsum_weights'`` of shape
        ``[input_shape[axis], size]`` and bias ``'bias'`` of shape
        ``[size]``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """
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
        """Apply the weighted-sum transform to the input tensor.

        Builds the layer on first call if it has not been built yet.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        demix : tf.Tensor
            ``nonlin(tensordot(x, w, axes=[[1], [0]]) + b_in)`` if
            ``self.nonlin`` is truthy, otherwise
            ``tensordot(x, w, axes=[[1], [0]]) + b_in``.

        """
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
        """Initialize a per-channel weighted-sum (depthwise) layer.

        Parameters
        ----------
        scope : str, optional
            Name scope / regularization-and-constraint lookup key for
            this layer. Defaults to 'wsum3d'.

        size : int, optional
            Number of output components per channel. Defaults to
            None.

        nonlin : callable, optional
            Nonlinearity applied to the layer's output. Defaults to
            ``tf.identity``.

        axis : int, optional
            Axis of the input tensor (excluding the last, channel,
            axis) to contract the weights against. Defaults to 1.

        specs : dict, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Defaults to ``{}``.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        self.scope = scope
        self.axis = axis
        super().__init__(
            size=size,
            nonlin=nonlin,
            specs=specs,
            **args)

    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'size'``,
            ``'nonlin'``, ``'axis'``, and ``'specs'``.

        """
        config = super(WeightSum3d, self).get_config()
        config.update({'scope': self.scope,
                       'size': self.size,
                       'nonlin': self.nonlin,
                       'axis': self.axis,
                       'specs': self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : WeightSum3d
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        cls.scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create the layer's per-channel weights and biases.

        Creates weight tensor ``'wsum_weights'`` of shape
        ``[input_shape[-1], input_shape[axis], size]`` and bias
        ``'bias'`` of shape ``[input_shape[-1], size]``, i.e. one
        weight matrix and bias vector per input channel.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """
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
        """Apply the per-channel weighted-sum transform to the input tensor.

        For each channel along the last axis of ``x``, contracts that
        channel's slice with its own weight matrix and adds its own
        bias, then stacks the per-channel results back along the last
        axis. Builds the layer on first call if it has not been built
        yet.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        result : tf.Tensor
            Per-channel weighted sums, stacked along the last axis.

        """
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
        """Initialize a per-channel squared weighted-sum (depthwise) layer.

        Parameters
        ----------
        scope : str, optional
            Name scope / regularization-and-constraint lookup key for
            this layer. Defaults to 'wsum3d'.

        size : int, optional
            Number of output components per channel. Defaults to
            None.

        nonlin : callable, optional
            Nonlinearity applied to the layer's output. Defaults to
            ``tf.identity``.

        axis : int, optional
            Axis of the input tensor (excluding the last, channel,
            axis) to contract the weights against (applied twice, see
            :meth:`call`). Defaults to 1.

        specs : dict, optional
            Regularization/constraint specification dict, see
            :class:`BaseLayer`. Defaults to ``{}``.

        **args : dict
            Additional keyword arguments passed to
            ``BaseLayer.__init__``.

        """
        self.scope = scope
        self.axis = axis
        super(SquareSum3d, self).__init__(size=size, nonlin=nonlin, specs=specs,
                                          **args)

    def get_config(self):
        """Return this layer's configuration for serialization.

        Returns
        -------
        config : dict
            Base config merged with ``'scope'``, ``'size'``,
            ``'nonlin'``, ``'axis'``, and ``'specs'``.

        """
        config = super(SquareSum3d, self).get_config()
        config.update({'scope': self.scope, 'size': self.size,
                       'nonlin': self.nonlin, 'axis': self.axis,
                       'specs': self.specs})
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a layer instance from its serialized config.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        layer : SquareSum3d
            A new layer instance built from ``config``.

        """
        nonlin_config = config.pop("nonlin")
        scope = config.pop("scope")
        nonlin = saving.deserialize_keras_object(nonlin_config)
        return cls(nonlin, **config)

    def build(self, input_shape):
        """Create the layer's per-channel weights and biases.

        Creates weight tensor ``'wsum_weights'`` of shape
        ``[input_shape[-1], input_shape[axis], size]`` (applied twice
        per channel, see :meth:`call`) and bias ``'bias'`` of shape
        ``[input_shape[-1], size]``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the layer's input.

        Returns
        -------
        None

        """

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
        """Apply the per-channel squared weighted-sum transform.

        For each channel along the last axis of ``x``, contracts that
        channel's slice with its own weight matrix twice in
        succession (rows, then columns) and adds its own bias, then
        stacks the per-channel results back along the last axis.
        Builds the layer on first call if it has not been built yet.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        training : bool, optional
            Currently unused. Defaults to None.

        Returns
        -------
        result : tf.Tensor
            Per-channel squared weighted sums, stacked along the last
            axis.

        """
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
    """Compute a soft (sigmoid + softmax) attention-weighted sum over rows.

    Learns a scalar attention weight per row via a sigmoid-activated
    dense layer, normalizes the weights with a softmax over the row
    axis, and returns the attention-weighted sum of ``x`` over that
    axis.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor whose second axis (axis 1) is attended over.

    Returns
    -------
    attention_output : tf.Tensor
        Attention-weighted sum of ``x`` over axis 1.

    """

    attention_weights = Dense(units=1, activation="sigmoid")(x)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)
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
