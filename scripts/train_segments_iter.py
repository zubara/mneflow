#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom training loop for keras_model that iterates over each sequence segment

@author: vranoug1
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_eager_execution()

import os
os.chdir('/m/home/home2/20/vranoug1/unix/OPM-BCI/mneflow-dev/scripts')
import mneflow

from mneflow import Dataset
import numpy as np
import gc
import keras_models
from keras_utils import plot_metrics, rmse, mse, r_square, soft_acc
from keras_utils import _speak, _track_metrics


# Dataset parameters
dpath = '/m/nbe/work/zubarei1/collabs/bcicomp_ecog/sub1/'
fname = 'sub1_comp.mat'

import_opt = dict(fs=1000,
                  savepath='./tfr/',
                  out_name='bcig_ecog5_v2',
                  input_type='seq',
                  overwrite=False,
                  val_size=0.2,
                  array_keys={'X': 'train_data', 'y': 'train_dg'},
                  picks=None,
                  target_picks=None,
                  target_type='float',
                  segment=500,
                  augment=True,
                  aug_stride=50,
                  transpose=True,
                  scale=True,
                  decimate=None,
                  bp_filter=False,
                  transform_targets=True)

meta = mneflow.produce_tfrecords([dpath+fname], **import_opt)

# batch the dataset according to that value
dataset = Dataset(meta, train_batch=1, class_subset=None, pick_channels=None, decim=None)

# load test dataset without batching
test_dataset = dataset._build_dataset(meta['test_paths'], n_batch=1)

# training parameters
n_epochs = 2
n_iter = 3000
patience = 3
eval_step = 250  # after how many samples do we want to evaluate
min_delta = 1e-6
l1_lambda = 3e-4
l2_lambda = 0
learn_rate = 3e-4

# model parameters
graph_specs = dict(n_ls=64,  # number of latent factors
                   filter_length=17,  # convolutional filter length
                   pooling=5,  # convlayer pooling factor
                   stride=5,  # stride parameter for pooling layer
                   padding='SAME',
                   model_path=meta['savepath'],
                   dropout=.25,
                   out_dim=meta['y_shape'],
                   axis=2,
                   l1=l1_lambda,
                   l2=l2_lambda,
                   rnn_units=meta['y_shape'],
                   rnn_dropout=0.25,
                   rnn_nonlin='tanh',
                   rnn_forget_bias=True,
                   rnn_seq=True)

# %% Set up model
optim = tf.keras.optimizers.Adam(learning_rate=learn_rate)
reg = tf.keras.regularizers.L1L2(l1=l1_lambda, l2=l2_lambda)
model = keras_models.VARCNNLSTM(graph_specs)
loss_f = mse

# %% Training loop - per single sequence segment
# Keep results for plotting
train_metrics = []
val_metrics = []

# how many sequences per Dataset, to avoid infinitely looping
train_elems = Dataset._get_n_samples(None, meta['train_paths'])
val_elems = Dataset._get_n_samples(None, meta['val_paths'])
for epoch in range(n_epochs):
    print('Start of epoch %d' % epoch)
    step = 0
    t_metrics = []
    v_metrics = []

    # Training loop - using batches of single sequence segments
    for x, y in dataset.train.shuffle(train_elems).take(train_elems):
        nb = x.shape[0].value - 1
        k = x.shape[1].value
        nseq = x.shape[2].value
        print('step', step, 'x shape', x.shape, 'y shape', y.shape)
        for kk in range(k):
            for s in range(nseq):
                _speak(s, step, 'step', n=100)

                with tf.GradientTape() as tape:

                    xn = x[nb, kk, s, :, :]
                    yn = y[nb, kk, s, :]

                    y_ = model(xn)
                    loss_value = loss_f(yn, y_)
                    r_ = [tf.keras.backend.flatten(w) for w in model.trainable_weights]
                    l1_l2 = tf.add_n([reg(w) for w in r_])
                    cost = loss_value + l1_l2

                # optim.minimize(cost, model.trainable_variables, name='minimize')
                grads = tape.gradient(cost, model.trainable_variables)
                grads_vars = zip(grads, model.trainable_variables)
                optim.apply_gradients(grads_vars, name='minimize')

                # Track progress
                tmp = [rmse(yn, y_), mse(yn, y_), r_square(yn, y_), soft_acc(yn, y_), cost, yn, y_]
                t_metrics = _track_metrics(t_metrics, tmp)

                step += 1
            # end of sequence segments - iterated over all segments

            # test on validation after every sequence
            print('stepping in validation after step: ', step)
            for vx, vy in dataset.val.shuffle(val_elems).take(val_elems):
                # print('vx shape', vx.shape, 'vy shape', vy.shape)
                for vs in range(vx.shape[2]):
                    _speak(vs, vs, 'vstep', n=100)

                    xn = vx[0, 0, vs, :, :]
                    yn = vy[0, 0, vs, :]
                    y_ = model(xn)
                    vloss_value = loss_f(yn, y_)
                    r_ = [tf.keras.backend.flatten(w) for w in model.trainable_weights]
                    l1_l2 = tf.add_n([reg(w) for w in r_])
                    vcost = vloss_value + l1_l2

                    # Track progress
                    tmp = [rmse(yn, y_), mse(yn, y_), r_square(yn, y_), soft_acc(yn, y_), vcost, yn, y_]
                    v_metrics = _track_metrics(v_metrics, tmp)

        # End single train sequence

    # end of epoch  - iterated over the whole dataset
    train_metrics.append(t_metrics)
    val_metrics.append(v_metrics)
    # end of epoch

# %% Test data
test_elems = Dataset._get_n_samples(None, meta['test_paths'])
test_metrics = []
for x, y in test_dataset.shuffle(test_elems).take(test_elems):
    # print('vx shape', vx.shape, 'vy shape', vy.shape)
    for s in range(x.shape[2]):
        _speak(s, s, 'tstep', n=100)

        xn = vx[0, 0, s, :, :]
        yn = vy[0, 0, s, :]
        y_ = model(xn)
        loss_value = loss_f(yn, y_)
        r_ = [tf.keras.backend.flatten(w) for w in model.trainable_weights]
        l1_l2 = tf.add_n([reg(w) for w in r_])
        cost = loss_value + l1_l2

        # Track progress
        tmp = [rmse(yn, y_), mse(yn, y_), r_square(yn, y_), soft_acc(yn, y_), vcost, yn, y_]
        test_metrics = _track_metrics(test_metrics, tmp)
test_metrics = [test_metrics]
# %% Plot
plot_metrics(train_metrics, title='Training')
plot_metrics(val_metrics, title='Validation')
plot_metrics(test_metrics, title='Test')

print('Custom training iteration over sequence segments completed')
