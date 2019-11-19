#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom training workflow for keras_model

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

# training parameters
n_iter = 10
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
# model = keras_models.VARCNNLSTM(graph_specs)
# model = keras_models.VARCNN(graph_specs)
model = keras_models.LSTM1L(graph_specs)
name = model.name.upper()
loss_f = mse

# %% Train model per sequence
train_m, val_m = model.train_per_sequence(dataset, loss_f, reg, optim, n_iter)
#  Test model
test_m = [model.eval_per_sequence(dataset, 'test', loss_f, reg)]
# %% Train model per segment
train_m, val_m = model.train_per_segment(dataset, loss_f, reg, optim, n_iter)
# Test model
test_m = [model.eval_per_segment(dataset, 'test', loss_f, reg)]
# %% Plot
plot_metrics(train_m, title=name+' Training', epochs=-1)
plot_metrics(val_m, title=name+' Validation', epochs=-1)
plot_metrics(test_m, title=name+' Test')

print('Custom training iteration over whole sequences completed')
