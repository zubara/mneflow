#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for mneflow models.

@author: Gavriela Vranou
"""
import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np


# %% --- Auxilliary functions ---
def _test_tfrecords(path='./', pattern=''):
    """Iterate over TFRecord files maching the pattern and checks for
    corrupted records."""
    import tensorflow as tf
    import glob

    CRED = '\033[91m'
    CGRE = '\033[92m'
    CEND = '\033[0m'
    total_files = 0
    error_files = 0
    train_files = sorted(glob.glob('%s*%s*.tfrecord' % (path, pattern)))
    for f_i, file in enumerate(train_files):
        try:
            total_files += sum(
                    [1 for _ in tf.python_io.tf_record_iterator(file)])
            print(CGRE + 'OK:\t', f_i, file + CEND)

        except IOError:
            print(CRED + 'ERROR:\t', f_i, file + CEND)
            total_files += 1
            error_files += 1
    print('Found %d files, %d raised an error.')


def get_targets(model, dataset, dset, steps):
    """Return true and model predicted targets from a mneflow.Dataset."""
    assert dset in ['train', 'val', 'test']
    data = getattr(dataset, dset)
    y_true = []
    y_pred = []
    for X, y in data.take(steps):
        y_ = model.predict(X)
        y_true.append(y)
        y_pred.append(y_)

    y_pred = np.argmax(np.vstack(y_pred), 1)
    y_true = np.argmax(np.vstack(y_true), 1)
    return y_true, y_pred


def report_results(model, dataset, dset, steps, event_names):
    """ Print a classification performance report on the desired dataset. """
    from sklearn.metrics import classification_report

    y_true, y_pred = get_targets(model, dataset, dset, steps)

    # Compare performance between Validation and Test data
    pset = ['Training', 'Validation', 'Test']
    ii = [idx for idx, elem in enumerate(pset) if dset in elem.lower()]
    dset = pset[ii[0]] if len(ii) else dset

    print('-------------------- '+dset+' ----------------------\n',
          classification_report(y_true, y_pred, target_names=event_names))


# %% --- Metrics ---
def rmse(y_true, y_pred, axis=0):
    """Root mean squared error (rmse)"""
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=axis))


def mse_weighted(labels, predictions, thres=0.3, ww=0.5):
    """Weighted mean squared error, weight applied according to threshold"""
    w = tf.where(tf.less_equal(labels, thres),
                 ww*tf.ones_like(labels, tf.float32),
                 tf.ones_like(labels, tf.float32))
    return tf.losses.mean_squared_error(labels, predictions, weights=w)


def mae_weighted(labels, predictions, thres=0.3, ww=0.5):
    """Weighted mean absolute error, weight applied according to threshold"""
    mae = tf.keras.losses.MeanAbsoluteError()
    w = tf.where(tf.less_equal(labels, thres),
                 ww*tf.ones_like(labels, tf.float32),
                 tf.ones_like(labels, tf.float32))
    return mae(labels, predictions, sample_weight=w)


def mse(y_true, y_pred, axis=0):
    """Mean squared error (mse)"""
    return K.mean(K.square(y_pred - y_true), axis=axis)


def r_square(y_true, y_pred, axis=0):
    """Coefficient of determination (R squared)"""
    SS_res = K.sum(K.square(y_true - y_pred), axis=axis)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=axis)), axis=axis)
    return (1 - SS_res/(SS_tot + K.epsilon()))


def soft_acc(y_true, y_pred, axis=0, dec=2):
    """Soft Accuracy for float targets, rounded to the desired decimal."""
    decim = 10**dec
    y_t = K.round(y_true*decim)/decim
    y_p = K.round(y_pred*decim)/decim
    return K.mean(K.equal(y_t, y_p), axis=axis)


# %%  --- Plot functions ---
def plot_output(model, dataset, dset, steps):
    """Plot the true output against the predicted output."""
    import matplotlib.pyplot as plt

    y_true, y_pred = get_targets(model, dataset, dset, steps)

    fig = plt.figure(figsize=(12, 8))
    plt.ylabel("Output", fontsize=14)
    plt.plot(y_true, 'b-', label='y_true')
    plt.plot(y_pred, 'r--', label='y_pred')
    plt.legend(loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_history(history, title='', nepochs=0, nend=None):
    """Plot the tracked metric projectory stored in the model.history object"""
    import matplotlib.pyplot as plt
    h = history.history
    k = list(h.keys())
    total_m = len(k)
    m = total_m // 2  # half are from validation
    nrows = m  # int(np.ceil(m / 3))
    ncols = 1  # int(np.ceil(m / nrows))

    # plt.subplots(nrows, ncols)
    plt.subplots(nrows, ncols, sharex=True)
    stop = nend if nend else len(history.epoch)
    start = nepochs if nepochs >= 0 else (stop + nepochs)
    xticks = np.arange(start, stop, dtype=int)

    for ii in range(m):
        t_label = k[ii]
        v_label = k[m + ii]
        t_values = h[t_label][start:stop]
        v_values = h[v_label][start:stop]

        plt.subplot(nrows, ncols, ii+1)
        plt.plot(xticks, t_values, 'blue')
        plt.plot(xticks, v_values, 'orange')
        plt.ylabel(t_label)
        if not ii:
            plt.title(title)
            plt.legend(['Train', 'Val'], loc='best')

    plt.xlabel('Epoch')
    plt.show()
