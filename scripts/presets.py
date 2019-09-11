#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility code: standardised workflow for evaluating model performance on
multiple datasets, with preset parameters.

Important: check the validity of dpath for each dataset.

Assumed file setup of D_PATH folder:
    . presets.py    The current file
    ./opm_emg       Contains OPM-EMG dataset 7-8 motions
    ./opm_emg/tfr   OPM-EMG meta TFR records are saved here
    ./plots         All figures are saved here
    ./squid_emg     Contains SQUID-EMG dataset of 5 motions
    ./squid_emg/tfr SQUID-EMG meta TFR records are saved here


@author: vranoug1
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import mneflow
from mneflow import dev, Dataset, Optimizer
import numpy as np
import os
import pickle
import mne

D_PATH = '/m/nbe/work/vranoug1/OPM-BCI/'


def rms(x, f_length=125):
    x = x**2
    cf = np.ones(f_length)/float(f_length)
    x = np.convolve(a=x, v=cf, mode='same')
    return np.sqrt(x)


def rescale(x):
    x -= x.min(keepdims=True)
    x /= x.std(keepdims=True)
    return x*1e-3


def sliding_std(x, f_length=1000):
    x = np.concatenate([np.zeros(f_length-1), x])
    nrows = x.size - f_length + 1
    n = x.strides[0]
    x2D = np.lib.stride_tricks.as_strided(x, shape=(nrows, f_length),
                                          strides=(n, n))
    return np.std(x2D, axis=1)


def split_val_test(X, Y):
    '''
    Splits Validation set into Validation and Test sets
    '''
    lab = np.unique(Y)
    idx_val = []
    idx_test = []
    for ii in lab:
        tmp = (Y == ii)
        tt = np.sum(tmp)//2
        idx = np.random.permutation(np.where(tmp)[0])
        idx_val.extend(idx[0:tt])

    idx_test = np.delete(range(0, len(Y)), idx_val)

    X_val = X[idx_val, :, :]
    Y_val = Y[tuple([idx_val])]
    X_test = X[idx_test, :, :]
    Y_test = Y[tuple([idx_test])]

    return X_val, Y_val, X_test, Y_test


# %%
def preset_data(preset, **args):
    "Quick load between different known datasets"

    assert preset in ['opm', 'emg', 'squid', 'mag', 'meg'], "Unknown preset parameter."

    if preset in ['opm', 'emg']:
        dpath = D_PATH + 'opm_emg/'
        raw_suff = '_wrist_mov.fif'
        data_id = 's01'  # s01 and s02 need to remove event 128 ('grasp')
        sensors = preset
        cont = args.get("cont", True)

        fname = dpath + data_id + raw_suff
        raw = mne.io.RawFIF(fname, preload=True, verbose=False)
        events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003,
                                 output='step')

        event_names = ['ext_wrist', 'flex_wrist', 'flex_index',
                       'flex_thumb', 'palm_down', 'palm_up', 'precision_grip']

        if data_id == 's03':
            event_names.append('grasp')
        elif data_id == 's02':  # s02 has an extra channel
            raw.info['bads'] = ['MEG1311']

        event_id = {v: 2**i for i, v in enumerate(event_names)}

        opm_names = ['MEG1111', 'MEG1121', 'MEG1131', 'MEG1141',
                     'MEG1211', 'MEG1221', 'MEG1231', 'MEG1241']
        emg_names = ['MISC005', 'MISC006', 'MISC007', 'MISC008', 'MISC011']

        opm_picks = mne.pick_channels(raw.info['ch_names'], opm_names)
        emg_picks = mne.pick_channels(raw.info['ch_names'], emg_names)

        for k in emg_picks:
            raw.info['chs'][k]['kind'] = 302

        # % preprocess raw file
        fmin = 1.
        fmax = 125.
        raw.notch_filter(np.arange(50, 251, 50), notch_widths=2, picks=opm_picks)
        raw = raw.filter(l_freq=fmin, h_freq=fmax, picks=opm_picks, method='iir')
        raw = raw.filter(l_freq=fmin, h_freq=None, picks=emg_picks, method='iir')
        raw = raw.apply_function(rms, picks=emg_picks)

        # extract (long) epochs
        reject = dict(mag=1e-10)  # {'mag': 1e-10}
        epochs = mne.epochs.Epochs(raw, events, event_id=event_id,
                                   tmin=-3., tmax=16, decim=2., detrend=None,
                                   reject=reject,
                                   preload=True,
                                   baseline=None)
        if args.get("dropbad", 0):
            epochs.drop_bad()
        epochs = epochs[event_names]

        if data_id == 's03':  # epochs 35-39 have signal only on MEG1141
            epochs = epochs[0:34]

        if sensors == 'opm':
            data = mne.epochs.rescale(epochs._data, times=epochs.times,
                                      baseline=(None, 2.), mode='zscore',
                                      picks=opm_picks)[:, opm_picks, :]
        else:
            data = mne.epochs.rescale(epochs._data, times=epochs.times,
                                      baseline=(None, 2.), mode='logratio',
                                      picks=emg_picks)[:, emg_picks, :]

        meta = dict(train_paths=[], val_paths=[], orig_paths=[], test_paths=[],
                    data_id=data_id, val_size=0, fs=None, savepath=dpath + 'tfr/')

        a = mneflow.utils.produce_labels(epochs.events[:, 2])
        labels, total_counts, meta['class_proportions'], meta['orig_classes'] = a
        data = data[..., 2501:]

        if cont:
            # Use all epochs, but portions of each will be split in Train/Val
            X_train, Y_train, X_val, Y_val = dev.preprocess_continuous(
                            [data], labels=labels, val_size=.2, overlap=True,
                            segment=250, stride=10)
        else:
            # separate epochs as Train/Val
            from sklearn.model_selection import train_test_split

            X_train,  X_val, Y_train, Y_val = train_test_split(data, labels,
                                                               test_size=0.2,
                                                               stratify=labels)
            X_train, Y_train = dev.preprocess_continuous(
                            [X_train], labels=Y_train, split=False,
                            overlap=True, segment=250, stride=10)
            X_val, Y_val = dev.preprocess_continuous(
                            [X_val], labels=Y_val, split=False, overlap=True,
                            segment=250, stride=10)

        if sensors == 'opm':
            X_train = dev.scale_to_baseline(X_train, baseline=None, crop_baseline=False)
            X_val = dev.scale_to_baseline(X_val, baseline=None, crop_baseline=False)

        # Split Validation set into validation and test set
        X_val, Y_val, X_test, Y_test = split_val_test(X_val, Y_val)

        meta['cont'] = cont
        meta['sensors'] = sensors
        meta['n_ch'], meta['n_t'] = X_train.shape[1:]
        meta['n_classes'] = len(meta['class_proportions'])
        meta['y_shape'] = Y_train.shape[1:]
        meta['task'] = 'classification'

        meta['train_paths'].append(''.join([meta['savepath'], meta['data_id'],
                                           '_train.tfrecord']))
        meta['val_paths'].append(''.join([meta['savepath'], meta['data_id'],
                                         '_val.tfrecord']))
        meta['test_paths'].append(''.join([meta['savepath'], meta['data_id'],
                                          '_test.tfrecord']))

        mneflow.utils._write_tfrecords(X_train, Y_train,
                                       meta['train_paths'][-1],
                                       task=meta['task'])
        mneflow.utils._write_tfrecords(X_val, Y_val,
                                       meta['val_paths'][-1],
                                       task=meta['task'])
        mneflow.utils._write_tfrecords(X_test, Y_test,
                                       meta['test_paths'][-1],
                                       task=meta['task'])

        with open(meta['savepath'] + data_id + '_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

    # %%
    elif preset in ['squid', 'mag', 'meg']:
        dpath = D_PATH + 'squid_emg/'
        raw_suff = '_supine_raw_sss.fif'
        data_id = 's02'
        sensors = 'mag'  # We are only interested in magnetometers
        cont = args.get("cont", True)
        # False will split training/validation from separate epochs

        fname = dpath + data_id + raw_suff
        raw = mne.io.RawFIF(fname, preload=True, verbose=False)
        events = mne.find_events(raw, stim_channel='STI101',
                                 min_duration=0.003, output='step')
        raw.pick_types(meg=sensors, eeg=False, emg=False)
        meg_picks = mne.pick_types(raw.info, meg='mag')
        # emg_picks = mne.pick_types(raw.info, meg=False, emg=True)
        # meg_names = [ raw.info['ch_names'][i] for i in meg_picks]
        # emg_names = [ raw.info['ch_names'][i] for i in emg_picks]

        event_names = ['flex_wrist', 'flex_index', 'flex_thumb', 'pistol']
        event_id = {v: 2**i for i, v in enumerate(event_names)}

        fmin = .1
        fmax = 90.
        raw.notch_filter(np.arange(50, 251, 50), notch_widths=2, picks=meg_picks)
        raw = raw.filter(l_freq=fmin, h_freq=fmax, picks=meg_picks, method='iir')
        # raw = raw.filter(l_freq=fmin, h_freq=None, picks=emg_picks, method='iir')
        # raw = raw.apply_function(rms, picks=emg_picks)
        # raw = raw.apply_function(rescale, picks=emg_picks)

        # extract (long) epochs
        epochs = mne.epochs.Epochs(raw, events, event_id=event_id,
                                   tmin=0., tmax=5, decim=2., detrend=None,
                                   baseline=(None, 2.),
                                   # reject={'grad': 4000e-12},
                                   preload=True)

        if args.get("dropbad", 0):
            epochs.drop_bad()
        epochs = epochs[event_names]

        data = epochs.get_data()
        data = dev.scale_to_baseline(data, baseline=(0, 1000), crop_baseline=False)
        data = data[..., 1001:]

        meta = dict(train_paths=[], val_paths=[], orig_paths=[], test_paths=[],
                    data_id=data_id, val_size=0, fs=None,
                    savepath=dpath + 'tfr/')
        a = mneflow.utils.produce_labels(epochs.events[:, 2])
        labels, total_counts, meta['class_proportions'], meta['orig_classes'] = a

        if cont:
            # Use all epochs, but portions of each will be split in Train/Val
            X_train, Y_train, X_val, Y_val = dev.preprocess_continuous(
                            [data], labels=labels, val_size=.2, overlap=True,
                            segment=250, stride=25)
        else:
            # separate epochs as Train/Val
            from sklearn.model_selection import train_test_split

            X_train,  X_val, Y_train, Y_val = train_test_split(data, labels,
                                                               test_size=0.2,
                                                               stratify=labels)
            X_train, Y_train = dev.preprocess_continuous(
                            [X_train], labels=Y_train, split=False,
                            overlap=True, segment=250, stride=25)
            X_val, Y_val = dev.preprocess_continuous(
                            [X_val], labels=Y_val, split=False, overlap=True,
                            segment=250, stride=25)

        # Split Val to Validation and Test
        X_val, Y_val, X_test, Y_test = split_val_test(X_val, Y_val)

        # from sklearn.svm import SVC
        # clf = SVC(kernel='rbf', C=100)
        # X_train = X_train.reshape(X_train.shape[0],-1)
        # X_val = X_val.reshape(X_val.shape[0],-1)
        # clf.fit(X_train,Y_train)
        # print(clf.score(X_val, Y_val))

        X_train = dev.scale_to_baseline(X_train, baseline=None,
                                        crop_baseline=False)
        X_val = dev.scale_to_baseline(X_val, baseline=None,
                                      crop_baseline=False)
        X_test = dev.scale_to_baseline(X_test, baseline=None,
                                       crop_baseline=False)

        meta['cont'] = cont
        meta['sensors'] = sensors
        meta['n_ch'], meta['n_t'] = X_train.shape[1:]
        meta['n_classes'] = len(meta['class_proportions'])
        meta['y_shape'] = Y_train.shape[1:]
        meta['task'] = 'classification'

        meta['train_paths'].append(''.join([meta['savepath'], meta['data_id'],
                                           '_train.tfrecord']))
        meta['val_paths'].append(''.join([meta['savepath'], meta['data_id'],
                                         '_val.tfrecord']))
        meta['test_paths'].append(''.join([meta['savepath'], meta['data_id'],
                                          '_test.tfrecord']))

        mneflow.utils._write_tfrecords(X_train, Y_train,
                                       meta['train_paths'][-1],
                                       task=meta['task'])
        mneflow.utils._write_tfrecords(X_val, Y_val,
                                       meta['val_paths'][-1],
                                       task=meta['task'])
        mneflow.utils._write_tfrecords(X_test, Y_test,
                                       meta['test_paths'][-1],
                                       task=meta['task'])

        with open(meta['savepath'] + data_id + '_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

    return meta, event_names


def squid_parameters(savepath='./tfr/'):
    '''
    Loads default parameters used on the SQUID-MAG dataset
    '''
    graph_specs = dict(n_ls=32,  # number of latent factors
                       filter_length=32,  # convolutional filter length
                       pooling=16,  # convlayer pooling factor
                       stride=8,  # stride parameter for pooling layer
                       padding='SAME',
                       model_path=savepath,
                       dropout=.25)
    optimizer_params = dict(l1_lambda=1e-3,
                            # l1_scope=['tconv', 'fc', 'dmx'],
                            l2_lambda=0,
                            # l2_scope=None,
                            learn_rate=3e-4,
                            task='classification',
                            class_weights=None)  # training batch size)
    train_params = dict(early_stopping=1, min_delta=1e-6, n_iter=1,
                        eval_step=250)

    return graph_specs, optimizer_params, train_params


def opm_parameters(savepath='./tfr/'):
    '''
    Loads default parameters used for OPMs on the OPM-EMG dataset
    '''
    graph_specs = dict(n_ls=32,  # number of latent factors
                       filter_length=32,  # convolutional filter length
                       pooling=16,  # convlayer pooling factor
                       stride=8,  # stride parameter for pooling layer
                       padding='SAME',
                       model_path=savepath,
                       dropout=.25)
    optimizer_params = dict(l1_lambda=1e-3,
                            l2_lambda=0,
                            learn_rate=3e-4,
                            task='classification',
                            class_weights=None)  # training batch size)
    train_params = dict(early_stopping=1, min_delta=5e-6, n_iter=1,
                        eval_step=250)
    return graph_specs, optimizer_params, train_params


def emg_parameters(savepath='./tfr/'):
    '''
    Loads default parameters used for EMGs on the OPM-EMG dataset
    '''
    graph_specs = dict(n_ls=32,  # number of latent factors
                       filter_length=32,  # convolutional filter length
                       pooling=16,  # convlayer pooling factor
                       stride=8,  # stride parameter for pooling layer
                       padding='SAME',
                       model_path=savepath,
                       dropout=.25)
    optimizer_params = dict(l1_lambda=1e-3,
                            l2_lambda=0,
                            learn_rate=3e-4,
                            task='classification',
                            class_weights=None)  # training batch size)
    train_params = dict(early_stopping=1, min_delta=5e-6, n_iter=1,
                        eval_step=250)
    return graph_specs, optimizer_params, train_params


def model_parameters(preset, savepath='./tfr/'):
    '''
    Meta function, extendable for different sensors.
    '''
    assert preset in ['opm', 'emg', 'squid'], "Unknown preset parameter."

    if preset in ['squid']:
        graph_specs, optimizer_p, train_p = squid_parameters(savepath)
    elif preset in ['opm']:
        graph_specs, optimizer_p, train_p = opm_parameters(savepath)
    else:
        graph_specs, optimizer_p, train_p = emg_parameters(savepath)

    return graph_specs, optimizer_p, train_p


# %%
def test_loop(meta, model, train_params, event_names, n=10, **args):
    '''
    Trains the provided model N times, evaluates it on the test data and plots
    Input: dataset metadata, event names, a model and its training parameters.

    '''
    from matplotlib import pyplot as plt
    from datetime import date
    from sklearn.metrics import classification_report

    dropbad = args.get("dropbad", False)
    tt = 'CONT' if meta['cont'] else 'DISC'
    subset_names = args.get("subset_names", event_names)
    savefig = args.get("savefig", True)
    if savefig:
        spath = args.get("savepath", './')

    for ii in range(n):
        if ii and savefig:
            plt.close()

        # Train model
        model.build()
        model.train(**train_params)

        # Evaluate performance
        model.update_log()
        print('%d. Finished %s:%s, DropBad:%s' % (ii, meta['sensors'].upper(), tt, str(dropbad)))
        model.evaluate_performance(data_path=meta['test_paths'], batch_size=None)

        # Compare performance between Validation and Test data
        tmp_pred, t_true = model.predict(data_path=meta['test_paths'])
        t_pred = np.argmax(tmp_pred, 1)

        tmp_pred, v_true = model.predict(data_path=meta['val_paths'])
        v_pred = np.argmax(tmp_pred, 1)

        print('-------------------- VALIDATION ----------------------\n',
              classification_report(v_true, v_pred, target_names=subset_names))
        print('----------------------- TEST -------------------------\n',
              classification_report(t_true, t_pred, target_names=subset_names))

        # Plot
        model.plot_cm(dataset='test', class_names=subset_names)

        if dropbad:
            figname = '%s_%s_drop_%d.png' % (meta['sensors'], tt, ii)
            title = '%s %s Confusion matrix DropBad' % ( meta['sensors'].upper(), tt)
        else:
            figname = '%s_%s_%d.png' % (meta['sensors'], tt, ii)
            title = '%s %s Confusion matrix' % ( meta['sensors'].upper(), tt)
        plt.title(title)
        h = plt.gcf()
        h.set_size_inches(6.4, 6)
        plt.tight_layout()

        if savefig:
            today = str(date.today())
            plt.savefig(spath + today + '_' + figname)


if __name__ == '__main__':
    # %%
    for preset in ['emg', 'opm', 'squid']:
        for dropbad in [True, False]:
            for cont in [True, False]:
                # % load data
                meta, event_names = preset_data(preset, dropbad=dropbad, cont=cont)

                # Initialise model parameters
                graph_specs, optimizer_params, train_params = model_parameters(preset, meta['savepath'])

                # Initialise dataset
                subset = None
                subset_names = event_names
                if subset:
                    subset_names = [v for i, v in enumerate(event_names) if i in subset]

                dataset = Dataset(meta, train_batch=200, class_subset=subset,
                                  pick_channels=None, decim=None)

                optimizer = Optimizer(**optimizer_params)
                model = mneflow.models.VARCNN(dataset, optimizer, graph_specs)

                # Loop
                test_loop(meta, model, train_params, event_names, n=1,
                          dropbad=dropbad, subset_names=subset_names,
                          savefig=True,
                          savepath=D_PATH+'plots/')

