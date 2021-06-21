# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:46:54 2020

@author: ipzub
"""
import os
os.chdir("C:\\Users\\ipzub\\projs\\mneflow")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import numpy as np
import mne
from mne.datasets import multimodal

import mneflow
mne.set_log_level(verbose='CRITICAL')

#fname_raw = os.path.join(multimodal.data_path(), 'multimodal_raw.fif')
#raw = mne.io.read_raw_fif(fname_raw)
#
#cond = raw.acqparser.get_condition(raw, None)
## get the list of condition names
#condition_names = [k for c in cond for k,v in c['event_id'].items()]
#epochs_list = [mne.Epochs(raw, **c) for c in cond]
#epochs = mne.concatenate_epochs(epochs_list)
#epochs = epochs.pick_types(meg='mag')
#print(epochs.info)
#%%


#Specify import options
import_opt = dict(savepath='C:\\data\\tfr\\',  # path where TFR files will be saved
                  out_name='mne_sample_epochs',  # name of TFRecords files
                  fs=600,
                  overwrite=False,
                  input_type='trials',
                  target_type='int',
                  n_folds=5,
                  picks={'meg':'grad'},
                  scale=True,  # apply baseline_scaling
                  crop_baseline=True,  # remove baseline interval after scaling
                  scale_interval=(0, 40),  # indices in time axis corresponding to baseline interval
                  test_set='holdout')


#write TFRecord files and metadata file to disk
#meta = mneflow.produce_tfrecords([epochs], **import_opt)
meta = mneflow.produce_tfrecords([], **import_opt)


dataset = mneflow.Dataset(meta, train_batch=100)
#%%
lf_params = dict(n_latent=64, #number of latent factors
                  filter_length=17, #convolutional filter length in time samples
                  nonlin = tf.nn.relu,
                  padding = 'SAME',
                  pooling = 5,#pooling factor
                  stride = 5, #stride parameter for pooling layer
                  pool_type='max',
                  model_path = import_opt['savepath'],
                  dropout = .5,
                  l1_scope = ["weights"],
                  l1=3e-3)

model = mneflow.models.LFCNN3(dataset, lf_params)
model.build()



#train the model for 10 epochs
model.train(n_epochs=30, eval_step=100, early_stopping=5)

#%%

model.evaluate(meta['test_paths'])
#%%
model.compute_patterns()

#%%
f1 = model.plot_patterns('Vectorview-grad', sorting='l2', class_names=condition_names)
#%%
f2 = model.plot_spectra(sorting='l2', norm_spectra='welch', class_names=condition_names)
#%%
f3 = model.plot_waveforms(sorting='l2', class_names=condition_names)
