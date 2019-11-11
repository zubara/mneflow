#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:49:38 2019

@author: vranoug1
"""
import tensorflow as tf
# tf.enable_eager_execution()

import numpy as np


# %% Auxilliary functions
def _get_metrics(vy, y_, vcost):
    tmp = [rmse(vy, y_), mse(vy, y_), r_square(vy, y_), soft_acc(vy, y_),
           vcost, vy, y_]
    return tmp


def _track_metrics(orig, new):
    m = len(new)
    new = np.asarray([tf.squeeze(ii).numpy() for ii in new]).reshape(m, -1)
    if len(orig):
        return np.concatenate((orig, new), axis=1)
    else:
        return new


def _speak(s, step, name='', n=100, t='.'):
    ''' Prints dots before it quacks '''
    if s % n:
        print(t, end='', flush=True)
    else:
        print(name, step)


def _get_factors(n):
    "Factorise integer value"
    assert (n > 0) and not n % 1, "Cannot factor non-positive integer value"
    return np.sort([
        factor for i in range(1, int(n**0.5) + 1) if n % i == 0
        for factor in (i, n//i)
    ])


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


def report_results(model, train, val, test, r_batch, event_names):
    # train, val, test = dataset.train, dataset.val, test_dataset
    from sklearn.metrics import classification_report

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


# %% Metrics
def rmse(y_true, y_pred, axis=-1):
    # root mean squared error (rmse) for regression
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=axis))


def mse(y_true, y_pred, axis=-1):
    # mean squared error (mse) for regression
    from tensorflow.keras import backend as K
    return K.mean(K.square(y_pred - y_true), axis=axis)


def r_square(y_true, y_pred, axis=-1):
    # coefficient of determination (R^2) for regression
    from tensorflow.keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred), axis=axis)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)), axis=axis)
    return (1 - SS_res/(SS_tot + K.epsilon()))


def soft_acc(y_true, y_pred, axis=-1, d=2):
    from tensorflow.keras import backend as K
    decim = 10**d
    y_t = K.round(y_true*decim)/decim
    y_p = K.round(y_pred*decim)/decim
    return K.mean(K.equal(y_t, y_p), axis=axis)


# %% Plot functions
def plot_metrics(t, title='', epochs=None):
    import matplotlib.pyplot as plt
    n_epoch = len(t)
    fig, axes = plt.subplots(6, sharex=True, figsize=(12, 8))
    fig.suptitle(title+':Raw Metrics')
    labels = []

    if not epochs:
        toprint = range(n_epoch)
    else:
        if isinstance(epochs, (list, tuple)):
            toprint = epochs
        elif isinstance(epochs, int):
            toprint = [epochs]

    for ii in toprint:
        eid = ii if ii >= 0 else (n_epoch+ii)
        labels.append('epoch %d' % eid)
        rmse, mse, rsquare, sacc, cost, yn, y_ = t[ii]

        axes[0].set_ylabel("RMSE", fontsize=14)
        axes[0].plot(rmse)

        axes[1].set_ylabel("MSE", fontsize=14)
        axes[1].plot(mse)

        axes[2].set_ylabel("R^2", fontsize=14)
        axes[2].plot(rsquare)

        axes[3].set_ylabel("Soft Accuracy", fontsize=14)
        axes[3].plot(sacc, '+')

        axes[4].set_ylabel("Cost", fontsize=14)
        axes[4].plot(cost)

        axes[5].set_ylabel("Output", fontsize=14)
        if len(labels) == 1:
            axes[5].plot(yn, 'r*')
        axes[5].plot(y_, '+')

        axes[5].set_xlabel("segment", fontsize=14)

    axes[0].legend(labels=labels, loc='upper left')
    axes[5].legend(labels=['y_true']+labels)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # Mean metrics
    if n_epoch > 1:
        fig, axes = plt.subplots(5, sharex=False, figsize=(12, 8))
        fig.suptitle(title+':Mean Metrics')
        tmp = np.asarray([np.mean(ii, axis=1) for ii in t])

        rmse, mse, rsquare, sacc, cost, _, _ = tmp.T
        axes[0].set_ylabel("RMSE", fontsize=14)
        axes[0].plot(range(n_epoch), rmse)

        axes[1].set_ylabel("MSE", fontsize=14)
        axes[1].plot(range(n_epoch), mse)

        axes[2].set_ylabel("R^2", fontsize=14)
        axes[2].plot(range(n_epoch), rsquare)

        axes[3].set_ylabel("Soft Accuracy", fontsize=14)
        axes[3].plot(range(n_epoch), sacc, '+')

        axes[4].set_ylabel("Cost", fontsize=14)
        axes[4].plot(range(n_epoch), cost)
        axes[4].set_xlabel("Epoch", fontsize=14)

        # n_seg = len(t[0][-1])
        # y_ = np.zeros(n_seg)
        # for ii in t:
        #     y_ += np.asarray(ii[-1])
        # y_ = y_/n_epoch
        # yn = t[0][-2]
        # axes[5].set_ylabel("Output", fontsize=14)
        # axes[5].plot(range(n_seg), yn, 'r*', label='y-true')
        # axes[5].plot(range(n_seg), y_, 'b+', label='y-pred')
        # axes[5].legend()
        # axes[5].set_xlabel("Segments", fontsize=14)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


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


def plot_cm(model, dataset, r_batch=1, class_names=None, normalize=False):

    """
    Plot a confusion matrix

    Parameters
    ----------

    dataset : str {'training', 'validation'}
            which dataset to use for plotting confusion matrix

    class_names : list of str, optional
            if provided subscribes the classes, otherwise class labels
            are used

    normalize : bool
            whether to return percentages (if True) or counts (False)
    """

    from matplotlib import pyplot as plt
    from sklearn.metrics import confusion_matrix
    import itertools

    try:
        y_true = _get_elem(dataset, r_batch)
    except Exception:
        y_true = _get_elem_g(dataset, r_batch)

    tmp = model.predict(dataset, steps=r_batch)
    y_pred = np.argmax(tmp, 1)
    f = plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    title = 'Confusion matrix: '
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    ax = f.gca()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.colorbar()
    if not class_names:
        class_names = np.arange(len(np.unique(y_true)))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylim(-0.5, tick_marks[-1]+0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    return f


# %% Custom training loops
def c_loss(model, x, y, task=None):
    from tensorflow.keras import losses
    if task in ['classification']:
        f = losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        f = losses.MeanSquaredError()
    y_ = model(x)
    return f(y_true=y, y_pred=y_)


def c_grad(model, x, y):
    with tf.GradientTape() as tape:
        loss_v = c_loss(model, x, y)
    return loss_v, tape.gradient(loss_v, model.trainable_variables)


def single_seq_train(model, optim, dataset, n_epochs=5, task=None):
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []

    for epoch in range(n_epochs):
        print('Start of epoch %d' % epoch)
        step = 0
        train_loss_avg = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.Accuracy()

        val_loss_avg = tf.keras.metrics.Mean()
        val_acc = tf.keras.metrics.Accuracy()

        # Training loop - using batches of 1 single sequence
        for x, y in dataset.train:
            print('step', step, 'x shape', x.shape, 'y shape', y.shape)
            # Optimize the model
            loss_value, grads = c_grad(model, x, y)
            optim.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            train_loss_avg(loss_value)  # Add current batch loss
            train_acc(y, model(x))

            step += 1

            for vx, vy in dataset.val:
                print('vx shape', vx.shape, 'vy shape', vy.shape)
                loss_value = c_loss(model, vx, vy)

                # Track progress
                val_loss_avg(loss_value)  # Add current batch loss
                val_acc(vy, model(vx))

        # End epoch
        train_loss_results.append(train_loss_avg.result())
        train_accuracy_results.append(train_acc.result())
        val_loss_results.append(val_loss_avg.result())
        val_accuracy_results.append(val_acc.result())

        print('Epoch %03d: Loss: %.3f, Accuracy: %.3f'
              % (epoch, train_loss_avg.result(), train_acc.result()))
        print('Seen so far: %s samples' % step)

    t = [(train_loss_results, train_accuracy_results),
         (val_loss_results, val_accuracy_results)]
    return t, model


def iterate_segments_train(model, optim, dataset, n_epochs=5, task=None):
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []

    for epoch in range(n_epochs):
        print('Start of epoch %d' % epoch)
        step = 0
        train_loss_avg = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.Accuracy()

        val_loss_avg = tf.keras.metrics.Mean()
        val_acc = tf.keras.metrics.Accuracy()

        # Training loop - using batches of single sequence segments
        for x, y in dataset.train:
            k = x.shape[0].value
            nseq = x.shape[1].value
            print('step', step, 'x shape', x.shape, 'y shape', y.shape)
            for kk in range(k):
                for s in range(nseq):
                    x0 = x[kk, s, :, :]
                    y0 = y[kk, s, :]
                    loss_value, grads = c_grad(model, x0, y0)
                    optim.apply_gradients(zip(grads, model.trainable_variables))
                    # Track progress
                    train_loss_avg(loss_value)  # Add current batch loss
                    train_acc(y0, model(x0))

                    step += 1
                # end of sequence segments - iterated over all segments
            # End single train sequence

            # Validation dataset
            for vx, vy in dataset.val:
                print('vx shape', vx.shape, 'vy shape', vy.shape)
                for vs in range(vx.shape[1]):
                    x0 = vx[0, vs, :, :]
                    y0 = vy[0, vs, :]
                    loss_value = c_loss(model, x0, y0)
                    # Track progress
                    val_loss_avg(loss_value)  # Add current batch loss
                    val_acc(y0, model(x0))

        # end of epoch  - iterated over the whole dataset
        train_loss_results.append(train_loss_avg.result())
        train_accuracy_results.append(train_acc.result())

        val_loss_results.append(val_loss_avg.result())
        val_accuracy_results.append(val_acc.result())
        # end of epoch

        print('Epoch %03d: Loss: %.3f, Accuracy: %.3f'
              % (epoch, train_loss_avg.result(), train_acc.result()))
        print('Seen so far: %s samples' % step)

    t = [(train_loss_results, train_accuracy_results),
         (val_loss_results, val_accuracy_results)]
    return t, model
