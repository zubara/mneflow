#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare mneflow vs keras model

@author: vranoug1
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import mneflow
from mneflow import dev, Dataset, Optimizer
import numpy as np
import gc
from presets import preset_data, get_subset, model_parameters
import keras_models
import presets

preset = 'squid'
dropbad = True
cont = False
n_iter = 10000
patience = 3
eval_step = 250  # after how many samples do we want to evaluate
min_delta = 5e-6

# Load data
meta, event_names = preset_data(preset, dropbad=dropbad, cont=cont)

# find total amount of samples for each dataset split
r_samples = Dataset._get_n_samples(None, meta['train_paths'])
t_batch = Dataset._get_n_samples(None, meta['test_paths'])
v_batch = Dataset._get_n_samples(None, meta['val_paths'])

# Factor the amount of training samples
tmp = keras_models._get_factors(r_samples)

# find the closest divisor to eval_stop
n_batch = tmp[np.argmin(abs(tmp - eval_step))]

# 1 in the case where the amount of samples is smaller than eval_step
r_batch = 1 if n_batch == r_samples else r_samples // n_batch

# pick a subset
subset = None
subset_names = get_subset(meta, subset=subset)

# batch the dataset according to that value
dataset = Dataset(meta, train_batch=n_batch, class_subset=subset,
                  pick_channels=None, decim=None)
# shuffle data
# dataset.train = dataset.train.shuffle(10000, reshuffle_each_iteration=True)

# load test dataset without batching
test_dataset = dataset._build_dataset(meta['test_paths'], n_batch=None)

print('loaded dataset')
# %% Initialise model parameters
graph_specs = dict(n_ls=64,  # number of latent factors
                   filter_length=32,  # convolutional filter length
                   pooling=56,  # convlayer pooling factor
                   stride=16,  # stride parameter for pooling layer
                   padding='SAME',
                   model_path=meta['savepath'],
                   dropout=.25,
                   out_dim=meta['n_classes'])

# mneflow model parameters
train_params = dict(early_stopping=patience, min_delta=min_delta,
                    n_iter=n_iter, eval_step=r_batch)

_, optimizer_params, _ = model_parameters(preset, meta['savepath'])
mne_opt = Optimizer(**optimizer_params)

# keras model parameters
keras_opt = tf.keras.optimizers.Adam()
keras_loss = 'sparse_categorical_crossentropy'
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                            min_delta=min_delta)

# build models
mne_model = mneflow.models.VARCNN(dataset, mne_opt, graph_specs)
mne_model.build()

keras_model = keras_models.VARCNN(graph_specs)
keras_model.compile(loss=keras_loss, optimizer=keras_opt, metrics=['accuracy'])

print('built models')

# %% fit mneflow model
mne_model.train(**train_params)

# Evaluate performance
mne_model.update_log()
mne_model.evaluate_performance(data_path=meta['test_paths'], batch_size=None)

# %% fit keras model
history = keras_model.fit(dataset.train, validation_data=dataset.val,
                          callbacks=[callback], epochs=n_iter,
                          steps_per_epoch=1, validation_steps=1)
results = keras_model.evaluate(test_dataset, steps=1, verbose=1)

# %% compare f1 scores
presets.val_test_results(mne_model, meta['train_paths'], meta['test_paths'], meta['val_paths'], subset_names)
keras_models.val_test_results(keras_model, dataset.train, dataset.val, test_dataset, r_batch, subset_names)

print('Finished both models.')
