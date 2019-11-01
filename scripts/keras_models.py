#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mneflow.Models implementation using keras models

@author: vranoug1
"""
import tensorflow as tf
# tf.enable_eager_execution()

import numpy as np
from keras_layers import DeMixing, VARConv, LSTMv1, Dense, DeConvLayer


class VARCNNLSTM(tf.keras.Model):
    """VAR-CNN-LSTM

    For details see [1].

    Parameters
    ----------
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

    rnn_units : int
        number of nodes in the LSTM layer

    rnn_dropout : float
        dropout for the LSTM layer

    rnn_nonlin : tf.activation
        activation function for the LSTM layer

    rnn_forget_bias : bool
        whether units forget bias

    References
    ----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
        """
    def __init__(self, specs, name='varcnn-lstm', **kwargs):
        """
        Parameters
        -----------
        specs : dict
                dictionary of model-specific hyperparameters.

        """
        super(VARCNNLSTM, self).__init__(name=name, **kwargs)
        self.scope = name
        self.specs = specs
        self._layers.remove(self.specs)  # Weirdly it's considered as a layer
        self.model_path = specs['model_path']
        self.rate = specs['dropout']
        self.out_dim = specs['out_dim']
        self.l1_l2 = tf.keras.regularizers.l1_l2(l1=self.specs['l1'],
                                                 l2=self.specs['l1'])

        self.demix = DeMixing(n_ls=self.specs['n_ls'],
                              axis=specs['axis'],
                              kernel_regularizer=self.l1_l2,
                              bias_regularizer=self.l1_l2)

        self.tconv1 = VARConv(scope="conv", n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'],
                              kernel_regularizer=self.l1_l2,
                              bias_regularizer=self.l1_l2)

        self.lstm = LSTMv1(scope="lstm",
                           size=self.specs['rnn_units'],
                           dropout=self.specs['rnn_dropout'],
                           nonlin=self.specs['rnn_nonlin'],
                           unit_forget_bias=self.specs['rnn_forget_bias'],
                           kernel_regularizer=self.l1_l2,
                           bias_regularizer=self.l1_l2,
                           return_sequences=self.specs['rnn_seq'])

        # self.fin_fc = Dense(size=self.out_dim, nonlin=tf.identity,
        #                     dropout=self.rate,
        #                     kernel_regularizer=self.l1_l2,
        #                     bias_regularizer=self.l1_l2)

# self.l1 = self.add_loss(specs['l1']*tf.reduce_sum(tf.abs(self.trainable_variables)))
# self.l2 = self.add_loss(specs['l2']*tf.nn.l2_loss(self.trainable_variables))

    def call(self, X):
        orig_dim = len(X.shape)
        # print(orig_dim)
        x0 = X
        if orig_dim > 4:
            # Input shape: [batch_index, k, n_seq, n_ch, time]
            x0 = tf.squeeze(X, axis=0)
        elif orig_dim == 4:
            k, n_seq, n_ch, time = X.shape    # [k, n_seq, n_ch, time]
            x0 = X
        elif orig_dim == 2:
            n_ch, time = X.shape
            x0 = tf.reshape(X, [1, 1, n_ch, time])  # [1, 1, n_ch, time]

        # print('input x0', x0.shape)
        x1 = self.demix(x0)                # [k, n_seq, time, 1, demix.size]
        # print('demix x1', x1.shape)
        x1a = tf.squeeze(x1, axis=0)       # [n_seq, time, 1, demix.size]
        # print('squeezed x1a', x1a.shape)
        x2 = self.tconv1(x1a)              # [n_seq, maxpool, tconv1.size]
        # print('varconv x2', x2.shape)
        col = tf.multiply(x2.shape[-1], x2.shape[-2])
        x2a = tf.reshape(x2, [-1, col])  # [n_seq, maxpool*tconv1.size]
        # print('reshaped x2a', x2a.shape)
        x2b = tf.expand_dims(x2a, axis=0)  # [k, n_seq, maxpool*tconv1.size]
        # print('expanded x2b', x2b.shape)
        y_ = self.lstm(x2b)                # [k, n_seq, y_shape]
        # print('lstm output y_', y_.shape)

        if orig_dim > 4:
            y_ = tf.expand_dims(y_, axis=0)  # [batch_index, k, n_seq, y_shape]
            # print('expanded y_', y_.shape)
        elif orig_dim == 2:
            y_ = tf.reshape(y_, [-1])
        # print('final output y_', y_.shape)
        return y_


class VARCNN(tf.keras.Model):

    """VAR-CNN

    For details see [1].

    Parameters
    ----------
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

    References
    ----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
        """
    def __init__(self, specs, name='var-cnn', **kwargs):
        """
        Parameters
        -----------
        specs : dict
                dictionary of model-specific hyperparameters.
        """
        super(VARCNN, self).__init__(name=name, **kwargs)
        self.scope = name
        self.specs = specs
        self.model_path = specs['model_path']
        self.rate = specs['dropout']
        self.out_dim = specs['out_dim']
        self.l1_l2 = tf.keras.regularizers.l1_l2(l1=self.specs['l1'],
                                                 l2=self.specs['l1'])

        self.demix = DeMixing(n_ls=self.specs['n_ls'],
                              kernel_regularizer=self.l1_l2,
                              bias_regularizer=self.l1_l2,
                              axis=2)
        self.tconv1 = VARConv(scope="conv", n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'],
                              kernel_regularizer=self.l1_l2,
                              bias_regularizer=self.l1_l2)

        self.fin_fc = Dense(size=self.out_dim, nonlin=tf.identity,
                            dropout=self.rate,
                            kernel_regularizer=self.l1_l2,
                            bias_regularizer=self.l1_l2)

    def call(self, X):
        x = self.demix(X)
        x = self.tconv1(x)
        x = self.fin_fc(x)
        return x


class VARDAE(tf.keras.Model):
    # Migrated from dev.py NOT TESTED
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

    def __init__(self, specs, name='var-cnn-autoencoder', **kwargs):
        """
        Parameters
        -----------
        specs : dict
                dictionary of model-specific hyperparameters.
        """
        super(VARDAE, self).__init__(name=name, **kwargs)
        self.scope = name
        self.l1_l2 = tf.keras.regularizers.l1_l2(l1=self.specs['l1'],
                                                 l2=self.specs['l1'])

        self.demix = DeMixing(n_ls=self.specs['n_ls'],
                              kernel_regularizer=self.l1_l2,
                              bias_regularizer=self.l1_l2)

        self.tconv1 = VARConv(scope="var-conv1",
                              n_ls=self.specs['n_ls'],
                              nonlin_out=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'],
                              kernel_regularizer=self.l1_l2,
                              bias_regularizer=self.l1_l2)

        self.tconv2 = VARConv(scope="var-conv2",
                              n_ls=self.specs['n_ls']//2,
                              nonlin_out=tf.nn.relu,
                              filter_length=self.specs['filter_length']//2,
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'],
                              kernel_regularizer=self.l1_l2,
                              bias_regularizer=self.l1_l2)

        self.encoding_fc = Dense(size=self.specs['out_dim'],
                                 nonlin=tf.identity,
                                 dropout=self.rate,
                                 kernel_regularizer=self.l1_l2,
                                 bias_regularizer=self.l1_l2)

        self.deconv = DeConvLayer(n_ls=self.specs['n_ls'],
                                  n_ch=self.dataset.h_params['n_ch'],
                                  n_t=self.dataset.h_params['n_t'],
                                  filter_length=self.specs['filter_length'],
                                  flat_out=False,
                                  kernel_regularizer=self.l1_l2,
                                  bias_regularizer=self.l1_l2)

    def call(self, X):
        x = self.demix(self.X)
        x = self.tconv1(x)
        x = self.tconv2(x)
        encoder = self.encoding_fc(x)
        decoder = self.deconv(encoder)
        return decoder


def test_model():
    # %%
    from mneflow import Dataset
    from presets import preset_data, get_subset, model_parameters
    from keras_utils import plot_history, report_results

    preset = 'squid'
    dropbad = True
    cont = False

    # Load data
    meta, event_names = preset_data(preset, dropbad=dropbad, cont=cont)

    # Initialise dataset
    subset = None
    subset_names = get_subset(meta, subset=subset)

    # find total amount of samples for each dataset split
    r_samples = Dataset._get_n_samples(None, meta['train_paths'])
    t_batch = Dataset._get_n_samples(None, meta['test_paths'])
    v_batch = Dataset._get_n_samples(None, meta['val_paths'])

    print(r_samples, t_batch, v_batch)

    n_batch = 250  # batch size

    # after how many batches to evaluate
    steps = 1 if n_batch >= r_samples else r_samples // n_batch

    # batch the train dataset
    dataset = Dataset(meta, train_batch=n_batch, class_subset=subset,
                      pick_channels=None, decim=None)

    # load test dataset without batching
    test_data = dataset._build_dataset(meta['test_paths'], n_batch=None)

    # Initialise model parameters
    graph_specs, _, _ = model_parameters(preset, meta['savepath'])
    graph_specs['out_dim'] = dataset.h_params['y_shape']

    opt = tf.keras.optimizers.Adam()
    loss = 'sparse_categorical_crossentropy'
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # build model
    model = VARCNN(graph_specs)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    history = model.fit(dataset.train, validation_data=dataset.val,
                        callbacks=[callback], epochs=2,
                        steps_per_epoch=steps, validation_steps=v_batch)

    results = model.evaluate(test_data, steps=t_batch, verbose=1)

    plot_history(history)

    report_results(dataset.train, dataset.val, test_data, steps, subset_names)


# %%
if __name__ == '__main__':
    print('Reloaded')
