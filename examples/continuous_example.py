#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:32:13 2022

@author: zubarei1
"""
import numpy as np
import tensorflow as tf
import os
os.chdir('C:\\Users\\ipzub\\projs\\mneflow\\')
import mneflow



# Dataset parameters
dpath = 'C:\\data\\bci4_ecog\\'
fname = 'sub1_comp.mat'

import scipy.io as sio
datafile = sio.loadmat(dpath + fname)
data = datafile['train_data'].T[np.newaxis,...]
print(data.shape)
events = datafile["train_dg"].T[np.newaxis,...]
print(events.shape)
#%% define transform_targets function

def transform_targets(targets):
    """
    This function specifies transformations to the input signals
    (e.g. accelerometer) to produce the (one-dimensional) target variable.

    Parametes:
    ----------
    targets: np.array
             measurements used to produce the target variable. For continous
             inputs targets should have dimentsions (n_trials, t, channels),
             where t is the same number of time samples as in the data X.
    Returns:
    --------
    out : np.array
          transformed target variables with dimentions (n_trials, 1)

    """
    out = []
    #as an illustration we take the mean of last 50 samples of the 1st channel
    #of the motioncaputre
    out = np.array([t[-50:, 0].mean(axis=0, keepdims=True) for t in targets])
    print(out.shape)
    ##alternative treatment
    # targets = np.mean(targets[:,:,0], axis=1)
    # print(targets.shape)
    # targets -= targets.min()
    # targets /= targets.max()
    # for t in targets:
    #     t = np.round(t,1)
    #     out.append([t])
    # out = np.array(out)
    # out = mneflow.utils.produce_labels(out, return_stats=False)
    # out = np.expand_dims(out, 1)
    # print("Target values: ", np.unique(out))
    assert out.ndim == 2
    return out
#%%

import_opt = dict(fs=1000,
                  savepath=dpath + '/tfr/',
                  out_name='cont_example',
                  input_type='continuous',
                  overwrite=True,
                  transform_targets=transform_targets,
                  target_type='float',
                  segment=625,
                  array_keys={"X":"train_data", "y":"train_dg"},
                  #augment=True,
                  scale = True,
#                  scale_y = True,
                  aug_stride=125,
                  save_as_numpy=True
                  )

meta = mneflow.produce_tfrecords((np.squeeze(data), events), **import_opt)

#%%
dataset = mneflow.Dataset(meta, train_batch = 100)

lf_params = dict(n_ls=32, #number of latent factors
                  filter_length=32, #convolutional filter length in time samples
                  pooling = 32,#pooling factor
                  nonlin = tf.nn.relu,
                  stride = 32, #stride parameter for pooling layer
                  padding = 'SAME',
                  dropout = 0.5,
                  model_path = import_opt['savepath'],
                  l1_lambda=3e-4,
                  l1_scope=[ 'fc', 'dmx'],
                  l2_scope=[ 'tconv'],
                  l2_lambda=3e-4,
                  pool_type='max') #path for storing the saved model

model = mneflow.models.LFCNN(dataset, lf_params)
model.build(loss='mae')

model.train(n_epochs=500, eval_step=50, early_stopping=10)
#model.update_log()
#%%
from matplotlib import pyplot as plt
y_true, y_pred = model.predict(meta['train_paths'])
plt.scatter(y_true, y_pred)
