#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:46:38 2019

@author: zubarei1
"""
import os
os.chdir('/m/nbe/work/zubarei1/mneflow/')
import mneflow

import scipy.io as sio
dpath = '/m/nbe/work/zubarei1/collabs/bcicomp_ecog/sub1/'
fname = 'sub1_comp.mat'

import_opt = dict(fs=1000, savepath='../tfr', out_name='bcig_ecog_test', input_type='seq',
                      overwrite=True, val_size=0.1,
                      array_keys={'X':'train_data', 'y':'train_dg'},
                      picks=None, target_picks=None,
                      target_type='float',
                      segment=500, augment=True, aug_stride=50, transpose=True,
                      scale=True,
                      decimate=None,
                      bp_filter = False,
                      transform_targets = True)

                      #) #validations set size set to 15% of all data

meta = mneflow.produce_tfrecords([dpath+fname], **import_opt)
#%%
optimizer_params = dict(l1_lambda=3e-6,learn_rate=3e-4, task='regression')

optimizer = mneflow.Optimizer(**optimizer_params)


dataset = mneflow.Dataset(meta, train_batch = 1, class_subset=None,
                          pick_channels=None, decim=None)


#lf_params = dict(n_ls=32, #number of latent factors
#              filter_length=16, #convolutional filter length in time samples
#              pooling = 8, #pooling factor
#              stride = 4, #stride parameter for pooling layer
#              padding = 'SAME',
#              dropout = .5,
#              model_path = import_opt['savepath']) #path for storing the saved model
#%%
#initialize the model using the dataset and optimizer objects, and the hyper-parameter dictionary
#model = mneflow.models.LFCNNR(dataset, optimizer, lf_params)

#this will initialize the iterators over the dataset,the computational graph and the optimizer
#model.build()

#%%
#from time import time
#start = time()
#model.train(n_iter=30,eval_step=10,min_delta=1e-6,early_stopping=5)
#stop = time() - start
#print('Trained in {:.2f}s'.format(stop))
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
#%%
#model.n_classes = 1
#model.compute_patterns(output='', norm_spect=True)
##%%
#model.plot_patterns(spectra=True, norm_spectra='ar')