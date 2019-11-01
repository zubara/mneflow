# -*- coding: utf-8 -*-
"""
Specifies utility functions.
"""
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import pickle
from operator import itemgetter
from mneflow.data import Dataset
from mneflow.optimize import Optimizer
from mne import filter as mnefilt#, not
from mne import epochs as mnepochs, pick_types
import csv
from mne import filter as mnefilt


def onehot(y):
    n_classes = len(set(y))
    out = np.zeros((len(y), n_classes))
    for i, ii in enumerate(y):
        out[i][ii] += 1
    return out.astype(int)


def load_meta(fname, data_id=''):

    """
    Loads a metadata file
    Parameters
    ----------

    fname : str
            path to TFRecord folder

    Returns
    -------
    meta : dict
        metadata file

    """
    with open(fname+data_id+'_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return meta


def leave_one_subj_out(meta, optimizer_params, graph_specs, model):
    """
    Performs a leave-one-out cross-validation such that on each fold one
    input .tfrecord file is used as a validation set.

    Parameters
    ----------
    meta : dict
            Dictionary containing metadata for initializing mneflow.Dataset.
            Normally meta is anoutput of produce_tfrecords function.

    optimizer_params : dict
            Dictionary of parameters for initializing mneflow.Opimizer.

    graph_specs : dict
            Dictionary of model-specific parameters.

    model : mneflow.models.Model
            Class of model to be used

    Returns
    -------
    results : list of dict
            List of dictionaries, containg final cost and performance estimates
            on each fold of the cross-validation
    """

    results = []
    optimizer = Optimizer(**optimizer_params)
    for i, path in enumerate(meta['orig_paths']):
        meta_loso = meta.copy()
        train_fold = [i for i, _ in enumerate(meta['train_paths'])]
        train_fold.remove(i)
        meta_loso['train_paths'] = itemgetter(*train_fold)(meta['train_paths'])
        meta_loso['val_paths'] = itemgetter(*train_fold)(meta['val_paths'])
        print('holdout subj:', path[-10:-9])
        dataset = Dataset(meta_loso, train_batch=200, class_subset=None,
                          pick_channels=None, decim=None)
        m = model(dataset, optimizer, graph_specs)
        m.build()
        m.train(n_iter=30000, eval_step=250, min_delta=1e-6, early_stopping=3)
        test_acc = m.evaluate_performance(path)
        print(i, ':', 'test_acc:', test_acc)
        results.append({'val_acc': m.v_acc, 'test_init': test_acc})
        # logger(m, results)
    return results


# def logger(savepath, h_params, params, results):
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
    """Perform global scaling based on a specified baseline.

    Subtracts the mean and divides by the standard deviation of the amplitude
    of all channels during the baseline interval. If input contains 306
    channels performs separate scaling for magnetometers and gradiometers.

    Parameters
    ----------
    X : ndarray
        data array with dimensions [n_epochs, n_channels, time].
    baseline : tuple of int, None
               baseline definition (in samples). If baseline == None the whole
               epoch is used for scaling.
    crop_baseline : bool
                    whether to crop the baseline after scaling is applied.

    Returns
    -------

    X : ndarray

    """

    if baseline is None:
        X0 = X.reshape([X.shape[0], -1])
        crop_baseline = False
    elif isinstance(baseline, tuple):
        interval = np.arange(baseline[0], baseline[1])
        X0 = X[..., interval].reshape([X.shape[0], -1])
        if crop_baseline:
            X = X[..., interval[-1]:]
    X0m = X0.mean(-1, keepdims=True)
    X0sd = X0.std(-1, keepdims=True)
    while X0m.ndim < X.ndim:
        X0m = np.expand_dims(X0m, -1)
        X0sd = np.expand_dims(X0sd, -1)
    # print(X0.shape)
    X -= X0m
    X /= X0sd

    return X


def make_example(X, y, input_type='iid', target_type='int'):
    if input_type in ['iid', 'trials']:
        feature = {}
        feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
        if target_type == 'int':
            feature['y'] = tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))
        else:
            y = y.astype(np.float32)
            feature['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))
        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))
    elif input_type == 'seq':
        sequence_length = len(X)
        # print('len x', sequence_length, 'shape x', X.shape)
        X_ = tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=s.flatten())) for s in X])
        y_ = tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=l.flatten())) for l in y])
        example = tf.train.SequenceExample(context={},
                                      feature_lists=tf.train.FeatureLists(feature_list={'X': X_, 'y':y_}))
        example.context.feature["length"].int64_list.value.append(sequence_length)
    #print(token.shape)
    return example


def _write_tfrecords(X_, y_, output_file, input_type='iid', target_type='int'):
    """Serialize and write datasets in TFRecords fromat

    Parameters
    ----------
    X_ : ndarray, shape (n_epochs, time_steps, n_channels, n_timepoints)
        (Preprocessed) data matrix.
    y_ : ndarray, shape (n_epochs,)
        Class labels.
    output_file : str
        Name of the TFRecords file.

    Returns
    -------
    TFRecord : TFrecords file.
    """
    writer = tf.io.TFRecordWriter(output_file)

    for X, y in zip(X_, y_):
        # print(len(X_), len(y_), X.shape, y.shape)
        X = X.astype(np.float32)
        # Feature contains a map of string to feature proto objects
        example = make_example(X, y, input_type=input_type, target_type=target_type)
        # Serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to the disk
        writer.write(serialized)
    writer.close()


def _split_sets(X, y, val=.1):
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
    return X_train, np.squeeze(y_train), X_val, np.squeeze(y_val)


def process_labels(y, scale=False, decimate=False, normalize=False,
                   transpose=False, transform=False, segment=False):
    """Preprocess target variables"""
    if transpose:
        y = np.swapaxes(y, -2, -1)

    if segment:
        y, _ = _segment(y, labels=None, segment_length=segment)
    if decimate:
        assert y.ndim == 3
        y = y[..., ::decimate]
    if normalize:
        y = scale_to_baseline(y, baseline=None, crop_baseline=False)
    y = np.mean(y**2, axis=-1)
    # print('y', y.shape)
#    if isinstance(transform, callable):
#        y = transform(y)

    return y


def import_continuous(inp, picks=None, target_picks=None, array_keys={'X': 'X', 'y': 'y'},
                      transpose=False, target_type='float'):
    """
    Returns
    -------
    data.ndim==3 [n_epochs, channels, times] and targets
    """

    if isinstance(inp, tuple) and len(inp) == 2:
        data, targets = inp
    elif isinstance(inp, str):
        fname = inp
        # print(fname[-3:])
        if fname[-3:] == 'mat':
            datafile = sio.loadmat(fname)

        if fname[-3:] == 'npz':
            datafile = np.load(fname)

        data = datafile[array_keys['X']]
    data = data.astype(np.float32)

    if data.ndim == 2:
        data = np.expand_dims(data, 0)
    if transpose:
        data = np.swapaxes(data, -1, -2)

    if target_type == 'float':
        if np.any(target_picks):
            targets = data[:, target_picks,:]
            data = np.delete(data, target_picks, axis=1)
            print('Extracting target variables from target_picks')
        else:
            targets = datafile[array_keys['y']]
            if targets.ndim == 2:
                targets = np.expand_dims(targets, 0)
            if transpose:
                targets = np.swapaxes(targets, -1, -2)
            print('Extracting target variables from {}'.format(array_keys['y']))

    if isinstance(picks, np.ndarray):
        data = data[:, picks, :]
    return data, targets


def import_epochs(inp, picks=None, array_keys={'X': 'X', 'y': 'y'}):
    """
    inputs : list, mne.epochs.Epochs, str
        list of mne.epochs.Epochs or strings with filenames. If input is a
        single string or Epochs object it is firts converted into a list.

    picks : ndarray of int, optional
        Indices of channels to pick for processing, if None all channels
        are used.

    array_keys : dict, optional
        Dictionary mapping {'X':'data_matrix','y':'labels'},
        where 'data_matrix' and 'labels' are names of the corresponding
        variables if your input is paths to .mat or .npz files.
        Defaults to {'X':'X', 'y':'y'}

    Returns
    -------
    data.ndim==3 [n_epochs, channels, times] and targets [n_epochs,...]

    """

    #  Import data and labels
    # for inp in inputs:
    if isinstance(inp, mnepochs.BaseEpochs):
        print('processing epochs')
        inp.load_data()
        data = inp.get_data()
        events = inp.events[:, 2]
        # fs = inp.info['sfreq']
        if isinstance(picks, dict):
            picks = pick_types(inp.info, **picks)
           # print('picks:', picks )
    elif isinstance(inp, tuple) and len(inp) == 2:
        print('importing from tuple!')
        data, events = inp
    elif isinstance(inp, str):
        fname = inp
        # print(fname[-3:])
        if fname[-3:] == 'fif':
            epochs = mnepochs.read_epochs(fname, preload=True,
                                          verbose='CRITICAL')
            if isinstance(picks, dict):
                picks = pick_types(epochs.info, **picks)
            events = epochs.events[:, 2]
            # fs = epochs.info['sfreq']
            data = epochs.get_data()
        else:
            if fname[-3:] == 'mat':
                datafile = sio.loadmat(fname)

            if fname[-3:] == 'npz':
                datafile = np.load(fname)

            data = datafile[array_keys['X']]
            events = datafile[array_keys['y']]
#        if not fs:
#            print('Specify sampling frequency')
        # return
    data = data.astype(np.float32)
    if isinstance(picks, np.ndarray):
        data = data[:, picks, :]
    print('data:', data.shape)
    return data, events


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
    class_proportions = {clss: cnt for clss, cnt in zip(inv[inds], counts)}
    orig_classes = {new: old for new, old in zip(inv[inds], classes)}
    if return_stats:
        return inv, total_counts, class_proportions, orig_classes
    else:
        return inv


def produce_tfrecords(inputs, fs, savepath, out_name, input_type='trials',
                      overwrite=True, savebatch=1, save_origs=False, val_size=0.2,
                      array_keys={'X': 'X', 'y': 'y'},
                      picks=None, target_picks=None,
                      combine_events=None, target_type='float',
                      segment=False, augment=False, aug_stride=50, transpose=False,
                      scale=False, scale_interval=None,
                      crop_baseline=False, decimate=False, bp_filter=False,
                      transform_targets=False):

    r"""
    Produces TFRecord files from input, applies (optional) preprocessing

    Calling this function will covnert the input data into TFRecords format
    that is used to effiently store and run Tensorflow models on the data.


    Parameters
    ----------


    savepath : str
        a path where the output TFRecord and corresponding metadata
        files will be stored.

    out_name :str
            filename prefix for the output files.

    savebatch : int
        number of input files per to be stored in the output TFRecord file.
        Deafults to 1.

    save_origs : bool, optinal
        If True, also saves the whole dataset in original order, e.g. for
        leave-one-subject-out cross-validation. Defaults to False.

    val_size : float, optional
        Proportion of the data to use as a validation set. Only used if
        shuffle_split = True. Defaults to 0.2.

    fs : float, optional
            Sampling frequency, required only if inputs are not mne.Epochs



    decimate : False, int, optional
        whether to decimate the input data (defaults to False).

    bp_filter : bool, tuple, optinal
        band pass filter. Tuple of int or NoneType.



    combine_events : dict, optional
        dictionary for combining or otherwise manipulating lables. SHould
        contain mapping {old_label: new_label}. If provided, but some
        old_labels are not specified in keys the corresponding epochs are
        discarded.



    task : 'classification', optional
        So far the only available task.



    overwrite : bool, optional
        Whether to overwrite the metafile if it already exists at the
        specified path.



    Returns
    -------
    meta : dict
        metadata associated with the processed dataset. Contains all the
        information about the dataset required for further processing with
        mneflow.
        Whenever the function is called the copy of metadata is also saved to
        savepath/meta.pkl so it can be restored at any time.



    Notes
    -----
    Pre-processing functions are implemented mostly for for convenience when
    working with array inputs.  When working with mne.epochs the use of
    the corresponding mne functions is preferrable.

    Examples
    --------
    meta = mneflow.produce_tfrecords(input_paths, \**import_opts)

    """

    if not os.path.exists(savepath):
        os.mkdir(savepath)
    if overwrite or not os.path.exists(savepath+out_name+'_meta.pkl'):

        meta = dict(train_paths=[], val_paths=[], test_paths=[], orig_paths=[],
                    data_id=out_name, val_size=0, savepath=savepath,
                    target_type=target_type, input_type=input_type)
        jj = 0
        i = 0
        meta['fs'] = fs
        if not isinstance(inputs, list):
            inputs = [inputs]
        for inp in inputs:
            if input_type=='trials':
                data, events = import_epochs(inp, picks=picks, array_keys=array_keys)
                if target_type == 'int':
                    events, total_counts, meta['class_proportions'], meta['orig_classes'] = produce_labels(events)
                    events = onehot(events)
                    #print(np.unique(events))

                    #meta['n_classes'] = len(meta['class_proportions'])
                x_train,  y_train, x_val, y_val = preprocess_epochs(data, events,
                                             scale=True, fs=fs, val_size=val_size,
                                             scale_interval=scale_interval,
                                             segment=segment, augment=augment,
                                             crop_baseline=crop_baseline,
                                             decimate=decimate, bp_filter=bp_filter)

                meta['y_shape'] = y_train[0].shape
            else:
                # TODO: continous classification
                data, events = import_continuous(inp, picks=picks,
                                                 target_picks=target_picks,
                                                 array_keys=array_keys,
                                                 transpose=transpose,
                                                 target_type='float')
                x_train,  y_train, x_val, y_val = preprocess_continuous(data, events,
                                                                       scale=scale,
                                                                       segment=segment,
                                                                       augment=augment,
                                                                       val_size=val_size,
                                                                       aug_stride=aug_stride,
                                                                       decimate=decimate,
                                                                       fs=fs,
                                                                       transform_targets=transform_targets,
                                                                       bp_filter=bp_filter,
                                                                       input_type=input_type)
                x_train = [np.moveaxis(ii, 0, -1) for ii in x_train]
                y_train = [np.moveaxis(ii, 0, -1) for ii in y_train]
                x_val = [np.moveaxis(ii, 0, -1) for ii in x_val]
                y_val = [np.moveaxis(ii, 0, -1) for ii in y_val]
                meta['y_shape'] = y_train[0].shape[-2]

            # print(x_train[0].shape, y_train[0].shape, meta['y_shape'])
            # meta['x_shape'] = x_train[0].shape
            if isinstance(x_train, list):
                meta['n_seq'], meta['n_ch'], meta['n_t'], n_epochs = x_train[0].shape
            else:
                 n_epochs,  meta['n_ch'], meta['n_t'] = x_train.shape
            print('Saving TFRecord# {}'.format(jj))

            # Split val to val and test
            x_test = [x_val[0][1::2, :, :, :]]
            y_test = [y_val[0][1::2, :, :]]
            x_val = [x_val[0][0::2, :, :, :]]
            y_val = [y_val[0][0::2, :, :]]

            meta['val_size'] += y_val[0].shape[1]
            meta['train_paths'].append(''.join([savepath, out_name,
                                                '_train_', str(jj),
                                                '.tfrecord']))
            _write_tfrecords(x_train, y_train, meta['train_paths'][-1],
                             input_type=input_type, target_type=target_type)
            meta['val_paths'].append(''.join([savepath, out_name,
                                              '_val_', str(jj),
                                              '.tfrecord']))
            _write_tfrecords(x_val, y_val, meta['val_paths'][-1],
                             input_type=input_type, target_type=target_type)

            meta['test_paths'].append(''.join([savepath, out_name,
                                               '_test_', str(jj),
                                               '.tfrecord']))
            _write_tfrecords(x_test, y_test, meta['test_paths'][-1],
                             input_type=input_type, target_type=target_type)

#                if save_origs:
#                    meta['orig_paths'].append(''.join([savepath, out_name,
#                                                       '_orig_', str(jj),
#                                                       '.tfrecord']))
#                    _write_tfrecords(X, y, meta['orig_paths'][-1], task=task)
            jj += 1
        with open(savepath+out_name+'_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

    elif os.path.exists(savepath+out_name+'_meta.pkl'):
        print('Metadata file found, restoring')
        meta = load_meta(savepath, data_id=out_name)
    return meta


def preprocess_epochs(data, events, val_size=.1, scale=False, fs=None, scale_interval=None,
                      crop_baseline=False, decimate=False, bp_filter=False,
                      picks=None, segment=False, augment=False):
    """
    scale : bool, optinal
        whether to perform scaling to baseline. Defaults to False.

    scale_interval : NoneType, tuple of ints or floats,  optinal
        baseline definition. If None (default) scaling is
        perfromed based on all timepoints of the epoch.
        If tuple, than baseline is data[tuple[0] : tuple[1]].
        Only used if scale == True.

    crop_baseline : bool, optinal
        whether to crop baseline specified by 'scale_interval'
        after scaling (defaults to False).


    Returns
    -------
    x_train, x_val - arrays of dimensions [n_epochs, n_seq, n_ch, n_t]
    y_train, y_val - arrays of dimensions [n_epochs, n_seq, n_targets]
    """

    if bp_filter:
        data = data.astype(np.float64)
        #print(data.shape)
        data = mnefilt.filter_data(data, fs, l_freq=bp_filter[0],
                                   h_freq=bp_filter[1],
                                   method='iir', verbose=False)
    if scale:
        data = scale_to_baseline(data, baseline=scale_interval,
                                 crop_baseline=crop_baseline)

    if isinstance(picks, np.ndarray):
        data = data[:, picks, :]

    if decimate:
        data = data[..., ::decimate]

    x_train, y_train, x_val, y_val = _split_sets(data, events, val=.1)


#    x_train = np.expand_dims(x_train,-1)
#    x_val = np.expand_dims(x_val,-1)
    # print('y_train:', y_train.shape)
    # = [np.expand_dims(s, -1) for s in sets]

    return x_train, y_train, x_val, y_val


def _combine_labels(labels, new_mapping):
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


def _segment(data, segment_length=200, augment=False, stride=25, input_type='iid'):
    """
    Parameters:
    -----------
    data : list of ndarrays
            data array of shape (n_epochs, n_channels, n_times)

    labels : ndarray
            array of labels (n_epochs,)

    segment_length : int or False
                    length of segment into which to split the data in time samples

    """
    # print('input_type:', input_type)
    x_out = []
    for xx in data:
        n_epochs, n_ch, n_t = xx.shape
        if augment:
            nrows = n_t - segment_length + 1
            a, b, c = xx.strides
            x4D = np.lib.stride_tricks.as_strided(xx,
                                                  shape=(n_epochs, n_ch, nrows, segment_length),
                                                  strides=(a, b, c, c))
            x4D = x4D[:, :, ::stride, :]
            x4D = np.swapaxes(x4D, 2, 1)
            if input_type != 'seq':
                x4D = x4D.reshape([n_epochs * x4D.shape[1], 1, n_ch, segment_length], order='C')
            # print('x4d:', x4D.shape)
            x_out.append(x4D)
        else:
            bins = np.arange(0, n_t+1, segment_length)[1:]
            xb = np.split(xx, bins, axis=-1)[:-1]
            x_out.append(np.concatenate(xb))
    # print([xxx.shape for xxx in x_out])
    if input_type != 'seq':
        return np.concatenate(x_out, 0)
    else:
        return(x_out)


def cont_split_indices(X, test_size=.1, test_segments=5):
    """
    X - 3d data array

    """
    raw_len = X.shape[-1]
    test_samples = int(test_size*raw_len//test_segments)
    interval = raw_len//(test_segments+1)
    data_intervals = np.arange(test_samples, raw_len-test_samples, interval)

    test_start = [ds + np.random.randint(interval - test_samples) for ds in data_intervals]

    test_indices = [(t_strt, t_strt+test_samples) for t_strt in test_start[:-1]]
    return test_indices


def partition(data, test_indices):
    if any(test_indices):
        if np.ndim(test_indices) == 1 and np.max(test_indices) < data.shape[0]:
            x_out = [data[test_indices,...], np.delete(data, test_indices, axis=0)][::-1]
        elif np.ndim(test_indices) == 2:
            bins = sorted(np.ravel(test_indices))
            x_out = np.split(data, bins, axis=-1) # data is a list of segmnets of different lengths
        else:
            print('Could not split the data, check test_indices!')
            return
        x_out = [xt - np.median(xt, axis=-1, keepdims=True) for xt in x_out]
        x_train = x_out[0::2]
        x_test = x_out[1::2]

        # print([xt.shape for xt in x_train])
        return x_train, x_test
    else:
        print('No test labels provided')
        return None


def preprocess_continuous(data, targets, scale=True, segment=200, augment=False,
                          val_size=0.1, aug_stride=10, transform_targets=True,
                          bp_filter=False, fs=None, decimate=None, input_type='iid'):
    """
    Returns
    -------
    x_train, x_val - arrays or lists of arrays of dimensions [n_epochs, n_seq, n_ch, n_t]
    y_train, y_val - arrays or lists of arrays of dimensions [n_epochs, n_seq, n_targets]
    """
    if bp_filter:
        data = data.astype(np.float64)
        data = mnefilt.notch_filter(data, fs, freqs=np.arange(50, 101, 50), notch_widths=1)

        data = mnefilt.filter_data(data, fs, l_freq=bp_filter[0],
                                   h_freq=bp_filter[1],
                                   method='iir', verbose=False)
    if transform_targets:
        poles = .02*np.ones(50)
        # print(targets.shape)
        targets_new = [np.convolve(targets[0,i,...], poles, mode='same') for i in range(5)]
        # print(targets_new[0].shape)
        targets = np.stack(targets_new)[None, ...]

        targets = mnefilt.filter_data(targets, sfreq=fs, l_freq=.05, h_freq=None,
                                      method='iir', verbose=False)

    if scale:
        data -= data.mean(-1, keepdims=True)
        data /= data.std()
        y_median = np.median(targets[0, :, :], axis=-1)
        # print(y_median.shape)
        # y_max = np.max(targets[0,0,...])
        q1, q5 = np.percentile(targets[0, ...], [5, 95], axis=-1)
        qrange = np.array(q5) - np.array(q1)
        # print(qrange.shape)

    if decimate:
        data = data[..., ::decimate]
        targets = targets[..., ::decimate]

    if val_size > 0:
        test_inds = cont_split_indices(data, test_size=val_size, test_segments=5)
        x_train, x_val = partition(data, test_inds)
        y_train, y_val = partition(targets, test_inds)

    if segment:
        # return
        x_train = _segment(x_train, segment_length=segment, augment=augment,
                           stride=aug_stride, input_type=input_type)

        x_val = _segment(x_val, segment_length=segment, augment=augment,
                         stride=aug_stride, input_type=input_type)
        y_train = _segment(y_train, segment_length=segment, augment=augment,
                           stride=aug_stride, input_type=input_type)
        y_val = _segment(y_val, segment_length=segment, augment=augment,
                         stride=aug_stride, input_type=input_type)
    if transform_targets:

        if input_type == 'seq':
            y_train = [np.mean(y_tr[..., 0, -aug_stride:], axis=-1, keepdims=True) for y_tr in y_train]
            y_val = [np.mean(y_v[..., 0, -aug_stride:], axis=-1, keepdims=True) for y_v in y_val]
            # y_train = [np.mean(y_tr[...,-25:],axis=-1) for y_tr in y_train]
            # y_val = [np.mean(y_v[...,-25:], axis=-1) for y_v in y_val]
        else:
            y_train = np.mean(y_train[..., -aug_stride:], axis=-1)
            y_val = np.mean(y_val[..., -aug_stride:], axis=-1)
            y_train -= y_median[None, None, :]
            y_val -= y_median[None, None, :]
            y_train /= qrange[None, None, :]
            y_val /= qrange[None, None, :]
#            y_train = np.squeeze(y_train)
#            y_val = np.squeeze(y_val)

        # y_train = np.max(y_train, -1, keepdims=True) - np.min()
    return x_train, y_train, x_val, y_val
