#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mneflow.Models implementation using keras models

@author: vranoug1
"""
import tensorflow as tf
# tf.enable_eager_execution()

import numpy as np
from keras_layers import DeMixing, VARConv, LSTMv1, Dense


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
    def __init__(self, specs, name='var-cnn', **kwargs):
        """
        Parameters
        -----------
        specs : dict
                dictionary of model-specific hyperparameters. Must include at
                least model_path - path for saving a trained model. See
                subclass definitions for details.

        """
        super(VARCNN, self).__init__(name=name, **kwargs)
        self.scope = name
        self.specs = specs
        self.model_path = specs['model_path']
        self.rate = specs['dropout']
        self.out_dim = specs['out_dim']

        self.demix = DeMixing(n_ls=self.specs['n_ls'])
        self.tconv1 = VARConv(scope="conv", n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])
        self.lstm = LSTMv1(scope="lstm",
                           size=self.specs['rnn_units'],
                           dropout=self.specs['rnn_dropout'],
                           nonlin=self.specs['rnn_nonlin'],
                           unit_forget_bias=self.specs['rnn_forget_bias'])

        self.fin_fc = Dense(size=self.out_dim, nonlin=tf.identity,
                            dropout=self.rate)

    def call(self, X):
        x = self.demix(X)
        x = self.tconv1(x)
        x, _ = self.lstm(x)
        x = self.fin_fc(x)
        return x


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
                dictionary of model-specific hyperparameters. Must include at
                least model_path - path for saving a trained model. See
                subclass definitions for details.

        """
        super(VARCNN, self).__init__(name=name, **kwargs)
        self.scope = name
        self.specs = specs
        self.model_path = specs['model_path']
        self.rate = specs['dropout']
        self.out_dim = specs['out_dim']

        self.demix = DeMixing(n_ls=self.specs['n_ls'])
        self.tconv1 = VARConv(scope="conv", n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        self.fin_fc = Dense(size=self.out_dim, nonlin=tf.identity,
                            dropout=self.rate)

    def call(self, X):
        x = self.demix(X)
        x = self.tconv1(x)
        x = self.fin_fc(x)
        return x


def plot_history(history):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(211)
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='best')
    plt.show()

    plt.subplot(212)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='best')
    plt.show()


def _get_factors(n):
    "Factorise integer value"
    assert (n > 0) and not n % 1, "Cannot factor non-positive integer value"
    return np.sort([
        factor for i in range(1, int(n**0.5) + 1) if n % i == 0
        for factor in (i, n//i)
    ])


def val_test_results(model, train, val, test, r_batch, event_names):
    # %
    # train, val, test = dataset.train, dataset.val, test_dataset
    from sklearn.metrics import classification_report

    def _get_elem(data, batches):
        y = []
        ii = 0
        for _, j in data:
            if ii >= batches:
                break
            else:
                y.extend(j)
                ii += 1

        return np.asarray(y)

    def _get_elem_g(dataset, batches):
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        y = []
        ii = 0
        with tf.compat.v1.Session() as sess:
            while True:
                if ii >= batches:
                    break
                else:
                    _, j = sess.run(next_element)
                    y.extend(j)
                    ii += 1

        return np.asarray(y)

    try:
        y_train = _get_elem(train, r_batch)
        y_val = _get_elem(val, 1)
        y_test = _get_elem(test, 1)
    except Exception:
        y_train = _get_elem_g(train, r_batch)
        y_val = _get_elem_g(val, 1)
        y_test = _get_elem_g(test, 1)

    # Compare performance between Validation and Test data
    tmp = model.predict(train, steps=r_batch)
    r_pred = np.argmax(tmp, axis=1)

    tmp = model.predict(val, steps=1)
    v_pred = np.argmax(tmp, axis=1)

    tmp = model.predict(test, steps=1)
    t_pred = np.argmax(tmp, axis=1)

    print('-------------------- TRAINING ----------------------\n',
          classification_report(y_train, r_pred, target_names=event_names))
    print('-------------------- VALIDATION ----------------------\n',
          classification_report(y_val, v_pred, target_names=event_names))
    print('----------------------- TEST -------------------------\n',
          classification_report(y_test, t_pred, target_names=event_names))


def test_model():
    # %%
    from mneflow import Dataset
    from presets import preset_data, get_subset, model_parameters

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

    # Factor the amount of training samples
    tmp = _get_factors(r_samples)

    # find the closest divisor to eval_stop
    eval_stop = 250  # after how many samples do we want to evaluate
    n_batch = tmp[np.argmin(abs(tmp - eval_stop))]

    # case where the amount of samples is smaller than the desired eval_stop
    r_batch = 1 if n_batch == r_samples else r_samples // n_batch

    # batch the dataset according to that value
    dataset = Dataset(meta, train_batch=n_batch, class_subset=subset,
                      pick_channels=None, decim=None)

    # load test dataset without batching
    test_dataset = dataset._build_dataset(meta['test_paths'], n_batch=None)

    # % Initialise model parameters
    graph_specs, _, _ = model_parameters(preset, meta['savepath'])
    graph_specs['out_dim'] = dataset.h_params['n_classes']

    opt = tf.keras.optimizers.Adam()
    loss = 'sparse_categorical_crossentropy'
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # % build model
    model = VARCNN(graph_specs)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    history = model.fit(dataset.train, validation_data=dataset.val,
                        callbacks=[callback], epochs=2000,
                        steps_per_epoch=r_batch, validation_steps=1)
    results = model.evaluate(test_dataset, steps=1, verbose=1)
    # %
    plot_history(history)

    val_test_results(dataset.train, dataset.val, test_dataset, r_batch)


# %%
if __name__ == '__main__':
    print('Reloaded')
