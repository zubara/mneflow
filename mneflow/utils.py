# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:12:48 2017
Read/ Preprocess data & labels/ Shuffle/  Split/ Scale/ Serialize/ Save
mne version = 0.15.2/ python2.7
@author: zubarei1
"""

import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import pickle
from operator import itemgetter
import mne
import csv


def load_meta(fname):
    """Loads a metadata file
    Parameters
    ----------

    fname : path to TFRecord folder
    """
    with open(fname+'meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return meta


def leave_one_subj_out(meta, params, specs, model, class_subset=None):
    """Performs a leave-one-subject out cross-validation"""
    results = []
    for i, path in enumerate(meta['orig_paths']):
        meta_loso = meta.copy()
        train_fold = [i for i, _ in enumerate(meta['train_paths'])]
        train_fold.remove(i)
        meta_loso['train_paths'] = itemgetter(*train_fold)(meta['train_paths'])
        meta_loso['val_paths'] = itemgetter(*train_fold)(meta['val_paths'])
        assert len(meta['train_paths']) == len(meta['val_paths'])
        assert len(meta_loso['train_paths']) != len(meta_loso['orig_paths'])
        print('holdout subj:', path[-10:-9])
        # TODO: Init model!!!
        # m = model(meta_loso, params, specs)
        m.build(class_subset=class_subset)
        m.train()
        test_acc = m.evaluate_performance(path, batch_size=None)
        print(i, ':', 'test_acc:', test_acc)
        #prt_test_acc, prt_logits = model.evaluate_realtime(path, step_size=params['test_upd_batch'])
        results.append({'val_acc': m.v_acc, 'test_init': test_acc})#, 'test_upd':np.mean(prt_test_acc), 'sid':h_params['sid']})
        #logger(savepath,h_params,params,results[-1])

    return results
#  def logger(savepath, h_params, params, results):
#    """Log model perfromance"""
#    #TODO: update logger for use meta
#    log = dict()
#    log.update(h_params)
#    log.update(params)
#    log.update(results)
#    for a in log:
#        if hasattr(log[a], '__call__'):
#            log[a] = log[a].__name__
#    header = ['architecture','data_id','val_acc','test_init', 'test_upd',
#              'n_epochs','eval_step','n_batch','y_shape','n_ch','n_t',
#              'l1_lambda','n_ls','learn_rate','dropout','patience','min_delta',
#              'nonlin_in','nonlin_hid','nonlin_out','filter_length','pooling',
#              'test_upd_batch', 'stride']
#    with open(savepath+'-'.join([h_params['architecture'],'training_log.csv']), 'a') as csv_file:
#        writer = csv.DictWriter(csv_file,fieldnames=header)
#        writer.writerow(log)


def scale_to_baseline(X, baseline=None, crop_baseline=False):
    """Perform scaling based on the specified baseline"""
    if baseline is None:
        interval = np.arange(X.shape[-1])
        crop_baseline = False
    elif isinstance(baseline, int):
        interval = np.arange(baseline)
    elif isinstance(baseline, tuple):
        interval = np.arange(baseline[0], baseline[1])
    X0 = X[:, :, interval]
    if X.shape[1] == 306:
        magind = np.arange(2, 306, 3)
        gradind = np.delete(np.arange(306), magind)
        X0m = X0[:, magind, :].reshape([X0.shape[0], -1])
        X0g = X0[:, gradind, :].reshape([X0.shape[0], -1])

        X[:, magind, :] -= X0m.mean(-1)[:, None, None]
        X[:, magind, :] /= X0m.std(-1)[:, None, None]
        X[:, gradind, :] -= X0m.mean(-1)[:, None, None]
        X[:, gradind, :] /= X0g.std(-1)[:, None, None]
    else:
        X0 = X0.reshape([X.shape[0], -1])
        X -= X.mean(-1)[..., None]
        X /= X0.std(-1)[:, None, None]
    if baseline and crop_baseline:
        X = X[..., interval[-1]:]
    return X


def write_tfrecords(X_, y_, output_file, task='classification'):
    """Serialize and write datasets in TFRecords fromat

    Parameters
    ----------
    X_ : ndarray, shape (n_epochs, n_channels, n_timepoints)
        (Preprocessed) data matrix.
    y_ : ndarray, shape (n_epochs,)
        Class labels.
    output_file : str
        Name of the TFRecords file.
    Returns
    -------
    TFRecord : TFrecords file.
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    for X, y in zip(X_, y_):
        X = X.astype(np.float32)
        # Feature contains a map of string to feature proto objects
        feature = {}
        feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
        if task == 'classification':
            feature['y'] = tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))
        elif task == 'ae':
            y = y.astype(np.float32)
            feature['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))
        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to the disk
        writer.write(serialized)
    writer.close()


def split_sets(X, y, val=.1):
    """Applies shuffle and splits the shuffled data

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_timepoints)
        (Preprocessed) data matrix.
    y : ndarray, shape (n_epochs,)
        Class labels.
    val : float from 0 to 1
        Name of the TFRecords file.
    Returns
    -------
    X_train, y_train, X_val, y_val : ndarray
    """
    shuffle = np.random.permutation(X.shape[0])
    val_size = int(round(val*X.shape[0]))
    X_val = X[shuffle[:val_size], ...]
    y_val = y[shuffle[:val_size], ...]
    X_train = X[shuffle[val_size:], ...]
    y_train = y[shuffle[val_size:], ...]
    return X_train, y_train, X_val, y_val


def produce_labels(y, return_stats=True):
    """Produces labels array from e.g. event (unordered) trigger codes

    Parameters
    ----------
    y : ndarray, shape (n_epochs,)
        array of trigger codes.

    Returns
    -------
    inv : ndarray, shape (n_epochs)
        ordered class labels.
    total_counts : int
    class_proportions : dict
        {new_class: proportion of new_class1 in the dataset}.
    orig_classes : dict
        {new_class:old_class}.
    """

    classes, inds, inv, counts = np.unique(y, return_index=True,
                                           return_inverse=True,
                                           return_counts=True)
    total_counts = np.sum(counts)
    counts = counts/float(total_counts)
    class_proportions = {clss: cnt for clss, cnt in zip(inds, counts)}
    orig_classes = {new: old for new, old in zip(inv[inds], classes)}
    if return_stats:
        return inv, total_counts, class_proportions, orig_classes
    else:
        return inv


def produce_tfrecords(inputs, savepath, out_name, overwrite=False,
                      savebatch=1,  save_origs=False, val_size=0.2, fs=None,
                      scale=False, scale_interval=None, crop_baseline=True,
                      decimate=False, bp_filter=False, picks=None,
                      combine_events=None, task='classification',
                      array_keys={'X': 'X', 'y': 'y'}):

    """Produces TFRecord files from input, applies (optional) preprocessing

    Calling this function will covnert the input data into TFRecords format
    that is used to effiently store and run Tensorflow models on the data.


    Parameters
    ----------
    inputs : list, mne.epochs.Epochs, str
        list of mne.epochs.Epochs or strings with filenames. If input is a
        single string or Epochs object it is firts converted into a list.

    savepath : str
        a path where the output TFRecord and corresponding metadata
        files will be stored.

    out_name :str
            filename prefix for the output files

    savebatch : int
        number of input files per to be stored in the output TFRecord file.
        Deafults to 1.

    save_origs : bool, optinal
        If True, also saves the whole dataset in original order, e.g. for
        leave-one-subject-out cross-validation. Defaults to False.

    val_size : float [0,1), optional
        Proportion of the data to use as a validation set. Only used if
        shuffle_split = True. Defaults to 0.2

    fs : float, int, optional
        Sampling frequency, required only if inputs are not mne.Epochs

    scale : bool, optinal
        whether to perform scaling to baseline. Defaults to False

    scale_interval : NoneType, tuple of ints or floats,  optinal
        baseline definition. If None (default) scaling is perfromed based on
        all timepoints of the epoch. If int, than baseline is defined as
        data[epoch_start : scale_interval], if tuple, than baseline is
        data[tuple[0] : tuple[1]]. Only used if scale = True
        #float for ecpohs?

    crop_baseline : bool, optinal
        whether to crop baseline specified by 'scale_interval' after scaling
        (defaults to False).

    decimate : False, int, optional
        whether to decimate the input data (defaults to False).

    bp_filter : bool, tuple, optinal
        band pass filter


    picks : ndarray of int, optional
        Indices of channels to pick for processing, if None all channels are
        used.


    task : 'classification', optional
        So far the only available task

    array_keys : dict, optional
        Dictionary mapping {'X':'data_matrix','y':'labels'},
        where 'data_matrix' and 'labels' are names of the corresponding
        variables if your input is paths to *mat or *npz files.
        Defaults to {'X':'X', 'y':'y'}

    overwrite : bool, optional
        Whether to overwrite the metafile if it already exists at the specified
        path

    Returns
    -------
    meta : dict
        metadata associated with the processed dataset. Contains all the
        information about the dataset required for further processing with
        mneflow.
        Whenever the function is called the copy of metadata is also saved to
        savepath/meta.pkl so it can be restored at any time



    Notes
    -----
    Pre-processing functions are implemented mostly for for convenience when
    working with array inputs.  When working with mne.epochs the use of
    the corresponding mne functions is preferrable.

    Examples
    --------
    1.Using mne.epochs
    import_opts = dict(savepath='path_to_output/', out_name='example',
                      picks={'meg':'grad'},scale=True,
                      crop_baseline=True, scale_interval=78,savebatch=1)
    meta = mneflow.produce_tfrecords(my_epochs,**import_opts)

    2.Using *.mat files
    input_paths = ['matlab_file_1.mat,'matlab_file_2.mat,'matlab_file_3.mat]
    import_opts = dict(savepath='path_to_output/', out_name='matlab_example',
                      scale=True, crop_baseline=True, scale_interval=(0,36),
                      savebatch=8, fs=500., bp_filter=(0.1,45.), decimate=4,
                      val_size=.15, array_keys={'X':'meg_data', 'y':'labels'},
                      savebatch=2)

    meta = mneflow.produce_tfrecords(input_paths,**import_opts)

    """

    if not os.path.exists(savepath):
        os.mkdir(savepath)
    if overwrite or not os.path.exists(savepath+'meta.pkl'):
        meta = dict(train_paths=[], val_paths=[], orig_paths=[],
                    data_id=out_name, val_size=0, task=task)
        jj = 0
        i = 0
        if not isinstance(inputs, list):
            inputs = [inputs]
        #  Import data and labels
        for inp in inputs:
            if isinstance(inp, mne.epochs.BaseEpochs):
                print('processing epochs')
                inp.load_data()
                data = inp.get_data()
                events = inp.events[:, 2]
                fs = inp.info['sfreq']
            elif isinstance(inp, tuple) and len(inp) == 2:
                data, events = inp
            elif isinstance(inp, str):
                fname = inp
                print(fname[-3:])
                if fname[-3:] == 'fif':
                    epochs = mne.epochs.read_epochs(fname, preload=True,
                                                    verbose='CRITICAL')
                    events = epochs.events[:, 2]
                    fs = epochs.info['sfreq']
                    data = epochs.get_data()
                else:
                    if fname[-3:] == 'mat':
                        datafile = sio.loadmat(fname)

                    if fname[-3:] == 'npz':
                        datafile = np.load(fname)

                data = datafile[array_keys['X']]
                events = datafile[array_keys['y']]
            if not fs:
                print('Specify sampling frequency')
                return

            #  IMPORT ENDS HERE!
            meta['fs'] = fs
            if isinstance(picks, np.ndarray):
                data = data[:, picks, :]
            # Preprocessing
            if bp_filter:
                data = mne.filter.filter_data(data, fs, l_freq=bp_filter[0],
                                              h_freq=bp_filter[1],
                                              method='iir', verbose=False)
            if scale:
                data = scale_to_baseline(data, scale_interval, crop_baseline)

            if decimate:
                data = data[..., ::decimate]
                meta['fs'] /= decimate

            if len(np.unique(events)) == 1:
                print('Events contain only one class!')
                break
            if task == 'classification':
                if combine_events:
                    events, keep_ind = combine_labels(events, combine_events)
                    data = data[keep_ind, ...]
                    events = events[keep_ind]

                labels, total_counts, meta['class_proportions'], meta['orig_classes'] = produce_labels(events)
                meta['n_classes'] = len(meta['class_proportions'])
                print('n_classes:', meta['n_classes'], ':', np.unique(labels))

            elif task == 'regression':
                print('Not Implemented')

            i += 1
            if i == 1:
                X = data
                y = labels
            elif i > 1:
                X = np.concatenate([X, data])
                y = np.concatenate([y, labels])

            meta['y_shape'] = y.shape[1:]

            if i % savebatch == 0 or jj*savebatch + i == len(inputs):
                print('data shape: ', X.shape)
                print('Saving TFRecord# {}'.format(jj))
                X = X.astype(np.float32)
                n_trials, meta['n_ch'], meta['n_t'] = X.shape
                X_train, y_train, X_val, y_val = split_sets(X, y, val=val_size)
                meta['val_size'] += len(y_val)
                meta['train_paths'].append(''.join([savepath, out_name,
                                                    '_train_', str(jj),
                                                    '.tfrecord']))
                write_tfrecords(X_train, y_train, meta['train_paths'][-1],
                                task=task)
                meta['val_paths'].append(''.join([savepath, out_name,
                                                 '_val_', str(jj),
                                                  '.tfrecord']))
                write_tfrecords(X_val, y_val, meta['val_paths'][-1], task=task)
                if save_origs:
                    meta['orig_paths'].append(''.join([savepath, out_name,
                                                       '_orig_', str(jj),
                                                       '.tfrecord']))
                    write_tfrecords(X, y, meta['orig_paths'][-1], task=task)
                jj += 1
                i = 0
                del X, y
            with open(savepath+'meta.pkl', 'wb') as f:
                pickle.dump(meta, f)

    elif os.path.exists(savepath+'meta.pkl'):
        meta = load_meta(savepath)
    return meta


def combine_labels(labels, new_mapping):
    """Combines labels

    Parameters
    ----------
    labels : ndarray
            label vector
    combine_dict : dict
            mapping {'new_label':[old_label1,old_label2]}
    """
    new_labels = 404*np.ones(len(labels), int)
    for old_label, new_label in new_mapping.items():
        ind = np.where(labels == old_label)[0]
        new_labels[ind] = new_label
    keep_ind = np.where(new_labels != 404)[0]
    return new_labels, keep_ind
