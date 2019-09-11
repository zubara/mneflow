#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:20:39 2019

@author: zubarei1
"""
import tensorflow as tf
from .models import Model
from .layers import DeMixing, VARConv, Dense, weight_variable, bias_variable
#from .utils import scale_to_baseline
import numpy as np


class VARDAE(Model):
    """ VAR-CNN

    For details see [1].

    Paramters:
    ----------
    var_params : dict

    n_ls : int
        number of latent components
        Defaults to 32

    filter_length : int
        length of spatio-temporal kernels in the temporal
        convolution layer. Defaults to 7

    stride : int
        stride of the max pooling layer. Defaults to 1

    pooling : int
        pooling factor of the max pooling layer. Defaults to 2

    References:
    -----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """

    def build_graph(self):
        self.scope = 'var-cnn-autoencoder'
        self.demix = DeMixing(n_ls=self.specs['n_ls'])

        self.tconv1 = VARConv(scope="var-conv1", n_ls=self.specs['n_ls'],
                              nonlin_out=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        self.tconv2 = VARConv(scope="var-conv2", n_ls=self.specs['n_ls']//2,
                              nonlin_out=tf.nn.relu,
                              filter_length=self.specs['filter_length']//2,
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        self.encoding_fc = Dense(size=self.specs['df'],
                                 nonlin=tf.identity,
                                 dropout=self.rate)

        encoder = self.encoding_fc(self.tconv2(self.tconv1(self.demix(self.X))))

        self.deconv = DeConvLayer(n_ls=self.specs['n_ls'],
                                  y_shape=self.dataset.h_params['y_shape'],
                                  filter_length=self.specs['filter_length'],
                                  flat_out=False)
        decoder = self.deconv(encoder)
        return decoder


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
                    out = out
                    if self.flat_out:
                        return tf.reshape(out, [-1, self.n_t*self.n_ch])
                    else:
                        print(out.shape)
                        return out
                except(AttributeError):
                    self.W = weight_variable((x.get_shape()[1].value,
                                              self.size))
                    self.b_in = bias_variable([self.size])
                    self.filters = weight_variable([self.n_t, 1])
                    self.b = bias_variable([self.size])
                    self.demixing = weight_variable((self.size, self.n_ch))
                    self.b_out = bias_variable([self.n_ch])
                    print(self.scope, 'init : OK')


def segment(data, labels, segment_length=200):
    """
    Parameters:
    -----------
    data : ndarray
            data array of shape (n_epochs, n_channels, n_times)

    labels : ndarray
            array of labels (n_epochs,)

    segment_length : int or False
                    length of segment into which to split the data in time samples

    """
    x_out = []
    y_out = []
    assert data.ndim == 3
    n_epochs, n_ch, n_t = data.shape
    bins = np.arange(0, n_t+1, segment_length)[1:]
    for x, y in zip(data, labels):
        #  split into non-overlapping segments
        xx = np.split(x, bins, axis=-1)[:-1]
        x_out.append(xx)
        y_out.append(np.repeat(y, len(xx)))
        #  print(y, y_out[-1])
    return np.concatenate(x_out), np.concatenate(y_out)


def augment(data, labels, segment_length, stride=25):
    """
    Parameters:
    -----------
    data : ndarray
            data array of shape (n_epochs, n_channels, n_times)

    labels : ndarray
            array of labels (n_epochs,)

    segment_length : int or False
                    length of segment into which to split the data in time samples

    stride : int
            stride in time samples for overlapping segments, defaults to 25

    """

    assert data.ndim == 3
    n_epochs, n_ch, n_t = data.shape
    nrows = n_t - segment_length + 1
    a, b, c = data.strides
    x4D = np.lib.stride_tricks.as_strided(data,
                                          shape=(n_epochs, n_ch, nrows, segment_length),
                                          strides=(a, b, c, c))
    x4D = x4D[:, :, ::stride, :]
    labels = np.tile(labels, x4D.shape[2])
    x4D = np.moveaxis(x4D, [2], [0])
    x4D = x4D.reshape([n_epochs * x4D.shape[0], n_ch, segment_length], order='C')
    return x4D, labels



def scale_to_baseline(X, baseline=None, crop_baseline=False, mode='standard'):
    """Perform global scaling based on a specified baseline.

    Subtracts the mean and divides by the standard deviation of the amplitude
    of all channels during the baseline interval. If input contains 306
    channels performs separate scaling for magnetometers and gradiometers.

    Parameters
    ----------
    X : ndarray
        data array with dimensions [n_epochs, n_channels, time].
    baseline : tuple of int, None
               baseline definition (in samples). If baseline == None the whole
               epoch is used for scaling.
    crop_baseline : bool
                    whether to crop the baseline after scaling is applied.

    Returns
    -------

    X : ndarray

    """
    if baseline is None:
        interval = np.arange(X.shape[-1])
        crop_baseline = False
    elif isinstance(baseline, tuple):
        interval = np.arange(baseline[0], baseline[1])
    X0 = X[:, :, interval]
    X0 = X0.reshape([X.shape[0], -1])
    if mode == 'standard':
        X -= X0.mean(-1)[:, None, None]
        X /= X0.std(-1)[:, None, None]
    elif mode == 'minmax':
        X -= X0.min(-1)[:, None, None]
        X /= X0.max(-1)[:, None, None]
    if baseline and crop_baseline:
        X = X[..., interval[-1]:]
    return X