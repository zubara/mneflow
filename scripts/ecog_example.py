#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:46:38 2019

@author: zubarei1
"""
import os
import numpy as np
os.chdir('/m/nbe/project/rtmeg/problearn/mneflow/')
import mneflow
import tensorflow as tf
import scipy.io as sio
dpath = '/m/nbe/work/zubarei1/collabs/bcicomp_ecog/sub1/'
fname = 'sub1_comp.mat'

def transform_targets(y):
    """
    Define transformation for target variables here and feed it to
    produce_tfrecords. Only used if target_type == 'signal'
        Parameters:
        -----------
    y : np.array
        target variable array with dimensions [n_trials, n_channels, n_times]

        Returns:
        --------
    y_out : np.array
        processed target variable with dimensions [n_samples, n_targets]
    """
    assert np.ndim(y) == 3, "Target variable has to be 3d array"
    # take only first channel and average over last 50 time samples
    y_out = [yy[0, -50:].mean(-1) for yy in y]
    # make sure that output is 2d array
    y_out = np.vstack(y_out)
    print(y_out.shape)
    if np.ndim(y_out) == 1:
        y_out = np.expand_dims(y_out, -1)
    return y_out

#%% Run import, preprocessing and produce TFRecords
import_opt = dict(fs=1000, savepath='../tfr', out_name='bcig_ecog',
                  input_type='continuous',
                  overwrite=False,
                  val_size=0.1,
                  array_keys={'X':'train_data', 'y':'train_dg'},
                  picks=None,
                  target_picks=None,
                  target_type='signal', #signal
                  segment=500,
                  aug_stride=50, # number of non-overlapping samples in augmentation
                  transpose=['X', 'y'],
                  scale=True,
                  decimate=None,
                  bp_filter=(3, 250), #band-pass filter
                  test_set = 'holdout',
                  transform_targets=transform_targets)

                      #) #validations set size set to 15% of all data

meta = mneflow.produce_tfrecords([dpath+fname], **import_opt)
#%% Initialize dataset and model using metadata
dataset = mneflow.Dataset(meta, train_batch=250, class_subset=None,
                          pick_channels=None, decim=None)

lf_params = dict(n_latent=32, #number of latent factors
                  filter_length=32, #convolutional filter length in time samples
                  nonlin = tf.nn.relu,
                  padding = 'SAME',
                  pooling = 4,#pooling factor
                  stride = 4, #stride parameter for pooling layer
                  pool_type='max',
                  model_path = import_opt['savepath'], #path for storing the saved model
                  dropout = .25,
                  l2_scope = ["fc" ,"lf_conv", "dmx"],
                  #l2_scope = ["lf_conv", "dmx"],
                  l2=3e-6)
                  #l2=0,
                  #maxnorm_scope=["demix"])

model = mneflow.models.LFCNN(dataset, lf_params)
model.build()

#%% Train and time
from time import time
start = time()
model.train(30, eval_step=100, val_batch=None, min_delta=1e-6,
              early_stopping=3)
stop = time() - start
print('Trained in {:.2f}s'.format(stop))
#%%
#from matplotlib import pyplot as plt
#import numpy as np
#
#y_pred, y_true = model.predict(meta['train_paths'])
#
##plt.hlines(np.median(y_true),0, len(y_true),)
#f, ax = plt.subplots(5,1)
#for  i, a in enumerate(ax):
#    a.plot(y_pred[:,i,0], alpha=.75)
#    a.plot(y_true[:,i,0],alpha=.5)
#plt.scatter(y_true,y_pred)
#plt.xlim(0,1)
#plt.ylim(-3,0)
#%% Plot stuff
#model.compute_patterns(meta['val_paths'])
#model.plot_spectra(norm_spect = 'welch')