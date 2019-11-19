#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:46:38 2019

@author: zubarei1
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
os.chdir('/m/nbe/project/rtmeg/problearn/mneflow/')
import mneflow



# Dataset parameters
dpath = '/m/nbe/work/zubarei1/collabs/bcicomp_ecog/sub1/'
fname = 'sub1_comp.mat'

import_opt = dict(fs=1000,
                  savepath='./tfr/',
                  out_name='bcig_ecog1_seq',
                  input_type='seq',
                  overwrite=True,
                  val_size=0.1,
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
                  transform_targets=True,
                  seq_length=10
                  )

meta = mneflow.produce_tfrecords([dpath+fname], **import_opt)
#%%
# batch the dataset according to that value
dataset = mneflow.Dataset(meta, train_batch=100, class_subset=None, pick_channels=None, decim=None)

# training parameters
optimizer_params = dict(l1_lambda=3e-6,learn_rate=3e-4, task='regression')

optimizer = mneflow.Optimizer(**optimizer_params)
# model parameters
graph_specs = dict(n_ls=32,  # number of latent factors
                   filter_length=16,  # convolutional filter length
                   pooling=4,  # convlayer pooling factor
                   stride=4,  # stride parameter for pooling layer
                   padding='SAME',
                   model_path=meta['savepath'],
                   dropout=.25,)

model = mneflow.models.LFLSTM(dataset,optimizer,graph_specs)
#model = mneflow.models.LFCNNR(dataset,optimizer,graph_specs)

model.build()

results = model.train(early_stopping=3, min_delta=5e-6, n_iter=50000,
                         eval_step=50)

#%%
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr
import numpy as np

y_pred, y_true = model.predict(meta['train_paths'])
y_pred = np.ravel(y_pred)
y_true = np.ravel(y_true)

r ,p = spearmanr(y_pred, y_true); print('spearman:', r, p)
r ,p = pearsonr(y_pred, y_true); print('pearson;',r, p)


#print('val')
#y_pred, y_true = model.predict(meta['val_paths'])
#y_pred = np.squeeze(y_pred)
#y_true = np.squeeze(y_true)
#r ,p = spearmanr(y_pred, y_true); print('spearman:', r, p)
#r ,p = pearsonr(y_pred, y_true); print('pearson', r, p)

#%%
f, ax = plt.subplots(1,1)
#for  i, a in enumerate(ax):
ax.plot(y_pred, alpha=.75)
ax.plot(y_true,alpha=.5)