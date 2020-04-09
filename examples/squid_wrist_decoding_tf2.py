#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:17:48 2019

@author: zubarei1
"""
#%%
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)

import tensorflow as tf
#tf.get_logger().setLevel('ERROR')
#tf.compat.v1.enable_eager_execution
import os
import numpy as np
os.chdir('/m/nbe/project/rtmeg/problearn/mneflow/')
import mneflow
print("Eager mode: ", tf.executing_eagerly())

from mneflow.keras_utils import plot_output, plot_history
import mne
path = '/m/nbe/project/rtmeg/problearn/wrist_decoding/'
#%%
#
#fnames = [path+'s02_wrist_'+str(i)+'_raw_tsss_mc.fif' for i in range(1,5)]
#
#epochs = []
#
#for fname in fnames:
#    raw = mne.io.RawFIF(fname, preload=True)
#    events = mne.find_events(raw, stim_channel='STI101', output='onset', min_duration=.003)
#    raw.pick_types(meg='grad', misc=False, eeg=False)
#    raw.notch_filter(np.arange(50.,101.,50.), notch_widths=1.)
#    raw = raw.filter(l_freq=1., h_freq=90.,  method='iir')
#    #
#    ep = mne.Epochs(raw, events, tmin=-1., tmax=1., decim=2., reject_by_annotation=False)
#    del raw
#    ep.drop_bad()
#    epochs.append(ep)
#epochs = mne.concatenate_epochs(epochs)

#%%

#data = epochs.get_data()
#events = epochs.events[:, 2]


import_opt = dict(fs=500, savepath=path+'/tfr/', out_name='s02_gradtf2', input_type='trials',
                      overwrite=False,
                      val_size=0.15,
                      target_type='int',
                      scale=True,
                      scale_interval = None,
                      crop_baseline=False,
                      decimate=None,
                      bp_filter = False)

meta = mneflow.produce_tfrecords([], **import_opt)

dataset = mneflow.Dataset(meta, train_batch=50, class_subset=None,
                          pick_channels=None, decim=None)
#%%
lf_params = dict(n_latent=25, #number of latent factors
                  filter_length=10, #convolutional filter length in time samples

                  #nonlin = tf.nn.relu,
                  padding = 'VALID',

                  pooling = 3,#pooling factor
                  stride = 3, #stride parameter for pooling layer
                  #pool_type='max',


                  model_path = import_opt['savepath'],
                  #dropout = .5,
                  l1_scope = ["fc"],
                  l2_scope = ["lf_conv"],
                  l1=3e-4,
                  l2=3e-2,
                  maxnorm_scope=["demix"]) #path for storing the saved model
#%%
model = mneflow.models.Deep4(dataset, lf_params)
model.build()
#%%
model.train(3, eval_step=10, val_batch=None, min_delta=1e-6,
              early_stopping=10)

#optimizer_params = dict(l1_lambda=3e-3, learn_rate=3e-4, task='classification')
#optim = tf.keras.optimizers.Adam(learning_rate=3e-4)
#loss_f = tf.compat.v1.losses.softmax_cross_entropy
#
## % builtin
#model.compile(loss=loss_f, optimizer=optim, metrics=['accuracy'])
#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=1)
##model.build()
#
##model.train(n_iter=30000,eval_step=250,min_delta=1e-6,early_stopping=5)
#
#history = model.fit(dataset.train, validation_data=dataset.val, epochs=30,
#                    steps_per_epoch=50,
#                    validation_steps=1, callbacks=[stop_early], verbose=1)
##
##%%
#model.update_log()
#model.compute_patterns(meta['train_paths'], output='filters')
##model.plot_patterns(sensor_layout='Vectorview-mag', sorting='best', spectra=False, scale=True)
##model.plot_waveforms(tmin= -1.)
##%%
##model.plot_cm()
#model.plot_spectra(fs=500, sorting='best', norm_spectra='welch', log=False)
#%%
#model.fake_evoked.save(meta['savepath']+meta['data_id']+'-ave.fif')