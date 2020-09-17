# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:12:48 2017
mne version = 0.15.2/ python3.7
@author: zubarei1
"""
import mne
import mneflow
from mneflow import dev, Dataset, Optimizer
import numpy as np
#import os
import pickle

dpath = '/m/nbe/work/zubarei1/opmbci/opm_emg/'
raw_suff = '_wrist_mov.fif'
data_id = 's02'
overwrite = True
sensors = 'emg'
#data_id = data_id + sensors

def rms(x, f_length=250):
    x = x**2
    cf = np.ones(f_length)/float(f_length)
    x = np.convolve(a=x, v=cf, mode='same')
    return np.sqrt(x)

# %% Import data

fname = dpath+data_id+raw_suff
raw = mne.io.RawFIF(fname, preload=True, verbose=False)
events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003,
                         output='step')

event_names = ['ext_wrist', 'flex_wrist', 'flex_index', 'flex_thumb',
               'palm_up', 'palm_dowm', 'pinch']

event_id = {v: 2**i for i,v in enumerate(event_names)}


opm_names = ['MEG1111', 'MEG1121', 'MEG1131', 'MEG1141',
             'MEG1211', 'MEG1221', 'MEG1231', 'MEG1241']

emg_names = ['MISC005', 'MISC006', 'MISC007', 'MISC008',
             'MISC011']

opm_picks = mne.pick_channels(raw.info['ch_names'], opm_names)
emg_picks = mne.pick_channels(raw.info['ch_names'], emg_names)

for k in emg_picks:
    raw.info['chs'][k]['kind'] = 302

# preprocess raw file
fmin = 5.
fmax = 90.
raw.notch_filter(np.arange(50, 251, 50), notch_widths=2, picks=opm_picks)
raw = raw.filter(l_freq=fmin, h_freq=fmax, picks=opm_picks, method='iir')
raw = raw.filter(l_freq=fmin, h_freq=None, picks=emg_picks, method='iir')
#  raw = raw.apply_function(rms, picks=emg_picks)

epochs = mne.epochs.Epochs(raw, events, event_id=event_id,
                           tmin=-3., tmax=15, decim=2., detrend=1,
                           reject={'mag': 1e-10},
                           preload=True,
                           baseline=None)
event_names.pop(3)
event_names.pop(-1)
epochs = epochs[event_names]
epochs.equalize_event_counts(event_names)


if sensors == 'opm':
    data = mne.epochs.rescale(epochs._data, times=epochs.times, baseline=(None,1.),
                       mode='mean', picks = opm_picks)[:,opm_picks,:]

elif sensors == 'emg':
#    data = mne.epochs.rescale(epochs._data, times=epochs.times, baseline=(None,1.),
#                       mode='zlogratio', picks = emg_picks)[:,emg_picks,:]
    data = epochs.get_data()[:,emg_picks,:]

meta = dict(train_paths=[], val_paths=[], orig_paths=[],
            data_id=data_id, val_size=0, fs=None,
            savepath=dpath + 'tfr/')
labels, total_counts, meta['class_proportions'], meta['orig_classes'] = mneflow.utils.produce_labels(epochs.events[:, 2])

shuffle = np.random.permutation(len(labels))
data = dev.scale_to_baseline(data, baseline=(0,2501), crop_baseline=True)
data = data[shuffle,:]
labels = labels[shuffle]
#labels, total_counts, meta['class_proportions'], meta['orig_classes'] = mneflow.utils.produce_labels(labels)

from sklearn.model_selection import train_test_split

#X, y = dev.augment(data, labels=labels, segment_length=250)
X_train,  X_val, Y_train, Y_val = train_test_split(data, labels, test_size=0.2, stratify=labels)

X_train, Y_train = dev.augment(X_train, labels=Y_train, segment_length=250)
X_val, Y_val = dev.segment(X_val, labels=Y_val, segment_length=250)
print(X_train.shape)


#if sensors == 'opm':
#X_train = dev.scale_to_baseline(X_train,baseline=None, crop_baseline=False)
#X_val = dev.scale_to_baseline(X_val,baseline=None, crop_baseline=False)

meta['n_ch'], meta['n_t'] = X_train.shape[1:]
meta['n_classes'] = len(meta['class_proportions'])
meta['y_shape'] = Y_train.shape[1:]
meta['task'] = 'classification'



meta['train_paths'].append(''.join([meta['savepath'], meta['data_id'],
                                   '_train.tfrecord']))
meta['val_paths'].append(''.join([meta['savepath'], meta['data_id'],
                                 '_val.tfrecord']))

mneflow.utils._write_tfrecords(X_train, Y_train,
                              meta['train_paths'][-1], task=meta['task'])
mneflow.utils._write_tfrecords(X_val, Y_val,
                              meta['val_paths'][-1], task=meta['task'])

with open(meta['savepath'] + data_id + '_meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

# %%
print(epochs)

subset = None

dataset = Dataset(meta, train_batch=200, class_subset=subset,
                          pick_channels=None, decim=None)
class_weights = dataset.class_weights()

optimizer_params = dict(l1_lambda=3e-5,
                        l1_scope=['tconv', 'fc', 'dmx'],
                        l2_lambda=0,
                        #l2_scope=None,
                        learn_rate=3e-4,
                        task='classification',
                        class_weights=None)  # training batch size)
for fl in [28]:
    for pooling in [50]:
        for n_ls in [32]:
            graph_specs = dict(n_ls=n_ls,  # number of latent factors
                               filter_length=fl,  # convolutional filter length
                               pooling=pooling,  # convlayer pooling factor
                               stride=25,  # stride parameter for pooling layer
                               padding='SAME',
                               model_path=meta['savepath'],
                               dropout=.5)

            optimizer = Optimizer(**optimizer_params)
            #
            model = mneflow.models.VARCNN(dataset, optimizer, graph_specs)
            #
            model.build()
            model.train(early_stopping=3, min_delta=5e-6, n_iter=30000,
                     eval_step=500)
            model.update_log()

if subset:
    subset_names = [v for i,v in enumerate(event_names) if i in subset]
    model.plot_cm(class_names=subset_names)
else:
    model.plot_cm(dataset='validation', class_names=event_names)

