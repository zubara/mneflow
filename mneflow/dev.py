#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:20:39 2019

@author: zubarei1
"""
import tensorflow as tf
from .models import Model
from .layers import DeMixing, VARConv, Dense, weight_variable, bias_variable
from .utils import scale_to_baseline
import numpy as np


class VARDAE(Model):
    """ VAR-CNN

    For details see [1].

    Paramters:
    ----------
    var_params : dict
                    {
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
                        }
    References:
    -----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """

    def _build_graph(self):
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


def preprocess_continuous(inputs, val_size=.1, overlap=False, segment=False,
                          stride=1, scale=False):
    """ Preprocess contious data

    Parameters:
    -----------
    inputs : list of ndarrays
            data to be preprocessed

    val_size : float
            proportion of data to use as validation set

    segment : int or False
            length of segment into which to split the data in time samples

    overlap : bool
            whether to use overlapping segments, False by default

    stride : int
            stride in time samples for overlapping segments, defaults to 1

    Returns:
    --------
    segments : tuple
            of size len(inputs)*2 traning and vaildation data split into
            (overlapping) segments

    Example:
    -------
    X_train, X_val = preprocess_continuous(X, val_size=.1, overlap=False,
                                           segment=False, stride=None)

    Returns two continuous data segments split into training and validation
    sets (90% and 10%, respectively). Validation set is defined as a single
    randomly picked segment of the data
    with length euqal to int(X.shape[-1]*val_size)

    X_train, X_val, Y_train, Y_val = train_test_split_cont([X,Y],val_size=.1,
    segment=500)

    Returns training and validation sets for two input arrays split into
    non-overlapping segments of 500 samples. This requires last
    dimentions of all inputs to be equal.


    X_train, X_val = preprocess_continuous(X, val_size=.1, overlap=True,
                                           segment=500, stride=25)

    Returns training and validation sets split into overlapping segments
    of 500 samples with stride of 25 time samples.

    """
    if not isinstance(inputs, list):
        inputs = list(inputs)
    split_datasets = train_test_split_cont(inputs, test_size=val_size)
    if overlap:
        segments = sliding_augmentation(split_datasets, segment=segment,
                                        stride=stride)
    elif segment:
        segments = segment_raw(inputs, segment, tile_epochs=True)
    else:
        segments = split_datasets
    if scale:
        segments = (scale_to_baseline(s, baseline=None,
                                      crop_baseline=False) for s in segments)
    return segments


def sliding_augmentation(datas, labels=None, segment=500, stride=1,
                         tile_epochs=True):
    """Return an image of x split in overlapping time segments"""
    output = []
    if not isinstance(datas, list):
        datas = [datas]
    for x in datas:
        while x.ndim < 3:
            x = np.expand_dims(x, 0)
        n_epochs, n_ch, n_t = x.shape
        nrows = n_t - segment + 1
        a, b, c = x.strides
        x4D = np.lib.stride_tricks.as_strided(x,
                                              shape=(n_epochs, n_ch, nrows, segment),
                                              strides=(a, b, c, c))
        x4D = x4D[:, :, ::stride, :]
        if tile_epochs:
            if labels:
                labels = np.tile(labels, x4D.shape[2])
            x4D = np.moveaxis(x4D, [2], [0])
            x4D = x4D.reshape([n_epochs*x4D.shape[0], n_ch, segment],
                              order='C')
        output.append(x4D)
#        print(x.shape,x4D.shape)
#        print(np.all(x[0,0,:segment]==x4D[0,0,...]))
#        print(np.all(x[0,1,:segment]==x4D[0,1,...]))
#        print(np.all(x[0,0,stride:segment+stride]==x4D[1,0,...]))
#        print(np.all(x[0,0,2*stride:segment+2*stride]==x4D[2,0,...]))
#        print(np.all(x[0,0,stride+segment_length:segment_length+stride*2] ==
#                     x4D[2,0,...]))
    if labels:
        output.append(labels)
    return output


def segment_raw(inputs, segment, tile_epochs=True):
    out = []
    if not isinstance(inputs, list):
        inputs = [inputs]
    raw_len = inputs[0].shape[-1]
    for x in inputs:
        assert x.shape[-1] == raw_len
        while x.ndim < 3:
            x = np.expand_dims(x, 0)
        orig_shape = x.shape[:-1]
        leftover = raw_len % (segment)
        print('dropping:', str(leftover), ':', leftover//2, '+',
              leftover-leftover//2)
        crop_start = leftover//2
        crop_stop = -1*(leftover-leftover//2)
        x = x[..., crop_start:crop_stop]
        x = x.reshape([*orig_shape, -1, segment])
        if tile_epochs:
            x = np.moveaxis(x, [-2], [0])
            x = x.reshape([orig_shape[0]*x.shape[0], *orig_shape[1:],
                           segment], order='C')
        out.append(x)
    return out


def train_test_split_cont(inputs, test_size):
    out = []
    if not isinstance(inputs, list):
        inputs = [inputs]
    raw_len = inputs[0].shape[-1]
    test_samples = int(test_size*raw_len)
    test_start = np.random.randint(test_samples//2,
                                   int(raw_len-test_samples*1.5))
    test_indices = np.arange(test_start, test_start+test_samples)
    for x in inputs:
        assert x.shape[-1] == raw_len
        x_test = x[..., test_indices]
        x_train = np.delete(x, test_indices, axis=-1)
        out.append(x_train)
        out.append(x_test)
    return out
