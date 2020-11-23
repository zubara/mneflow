# -*- coding: utf-8 -*-
"""
Specifies utility functions.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import os
import pickle
import warnings
from operator import itemgetter

import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import scipy.io as sio

import mne
#from mne import epochs as mnepochs, pick_types

from mneflow.data import Dataset
from mneflow.optimize import Optimizer


def _onehot(y, n_classes=False):
    if not n_classes:
        """Create one-hot encoded labels."""
        n_classes = len(set(y))
    out = np.zeros((len(y), n_classes))
    for i, ii in enumerate(y):
        out[i][ii] += 1
    return out.astype(int)


def _load_meta(fname, data_id=''):
    """Load a metadata file.

    Parameters
    ----------
    fname : str
        Path to TFRecord folder

    Returns
    -------
    meta : dict
        Metadata file

    """
    with open(fname+data_id+'_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return meta


def leave_one_subj_out(meta, optimizer_params, graph_specs, model):
    """Perform a leave-one-out cross-validation.

    On each fold one input .tfrecord file is used as a validation set.

    Parameters
    ----------
    meta : dict
        Dictionary containing metadata for initializing mneflow.Dataset.
        Normally meta is an output of produce_tfrecords function.

    optimizer_params : dict
        Dictionary of parameters for initializing mneflow.Optimizer.

    graph_specs : dict
        Dictionary of model-specific parameters.

    model : mneflow.models.Model
        Class of model to be used

    Returns
    -------
    results : list of dict
        List of dictionaries, containing final cost and performance
        estimates on each fold of the cross-validation.
    """
    results = []
    optimizer = Optimizer(**optimizer_params)
    for i, path in enumerate(meta['test_paths']):
        meta_loso = meta.copy()

        train_fold = [jj for jj, _ in enumerate(meta['train_paths'])]
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
        results.append({'val_acc': m.v_acc, 'test_acc': test_acc})

        # logger(m, results)
    return results


def scale_to_baseline(X, baseline=None, crop_baseline=False):
    """Perform global scaling based on a specified baseline.

    Subtracts the mean and divides by the standard deviation of the
    amplitude of all channels during the baseline interval. If input
    contains 306 channels, performs separate scaling for magnetometers
    and gradiometers.

    Parameters
    ----------
    X : ndarray
        Data array with dimensions [n_epochs, n_channels, time].

    baseline : tuple of int, None
        Baseline definition (in samples). If baseline == None the whole
        epoch is used for scaling.

    crop_baseline : bool
        Whether to crop the baseline after scaling is applied.

    Returns
    -------
    X : ndarray
        Scaled data array.

    """
    #X = X_.copy()

    if baseline is None:
        print("No interval specified, using the whole epochs")
        interval = np.arange(X.shape[-1])
    elif isinstance(baseline, tuple):
        print("Scaling to interval {:.1f} - {:.1f}".format(*baseline))
        interval = np.arange(baseline[0], baseline[1])
    X0m = X[..., interval].mean(axis=(1,2), keepdims=True)
    X0sd = X[..., interval].std(axis=(1,2), keepdims=True)

    X -= X0m
    X /= X0sd
    print("Scaling Done")
    return X


def _make_example(X, y, target_type='int'):
    """Construct Example proto object from data / target pairs."""

    feature = {}
    feature['X'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=X.flatten()))

    if target_type == 'int':
        feature['y'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=y.flatten()))
    elif target_type in ['float', 'signal']:
        y = y.astype(np.float32)
        feature['y'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=y.flatten()))
    else:
        raise ValueError('Invalid target type.')

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def _write_tfrecords(X_, y_, output_file, target_type='int'):
    """Serialize and write datasets in TFRecords format.

    Parameters
    ----------
    X_ : list of ndarrays
        (Preprocessed) data matrix.
        len = `n_epochs`, shape = `(time_steps, n_channels, n_timepoints)`

    y_ : list of ndarrays
        Class labels.
        len =  `n_epochs`, shape = `y_shape`

    output_file : str
        Name of the TFRecords file.
    """
    writer = tf.io.TFRecordWriter(output_file)

    for X, y in zip(X_, y_):
        # print(len(X_), len(y_), X.shape, y.shape)
        X = X.astype(np.float32)
        # Feature contains a map of string to feature proto objects
        example = _make_example(X, y, target_type=target_type)
        # Serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to the disk
        writer.write(serialized)
    writer.close()


def _split_sets(X, y, val=.1):
    """Apply shuffle and split the shuffled data.

    Parameters
    ----------
    X : ndarray
        (Preprocessed) data matrix.
        shape (n_epochs, n_channels, n_timepoints)

    y : ndarray
        Class labels.
        shape (n_epochs,)

    val : float from 0 to 1
        Name of the TFRecords file.

    Returns
    -------
    X_train, y_train, X_val, y_val : ndarray
        Pairs of data / targets split in Training and Validation sets.
    """
    assert (val > 0) and (val < 1), "Invalid val ratio."

    while y.ndim < 2:
        y = np.expand_dims(y, -1)

    shuffle = np.random.permutation(X.shape[0])
    val_size = int(round(val*X.shape[0]))
    X_val = X[shuffle[:val_size], ...]
    y_val = y[shuffle[:val_size], ...]
    X_train = X[shuffle[val_size:], ...]
    y_train = y[shuffle[val_size:], ...]
    # return X_train, np.squeeze(y_train), X_val, np.squeeze(y_val)
    return X_train, y_train, X_val, y_val


def import_data(inp, picks=None, target_picks=None,
                array_keys={'X': 'X', 'y': 'y'},
                transpose=False):
    """Import epoch data into `X, y` data/target pairs.

    Parameters
    ----------
    inp : list, mne.epochs.Epochs, str
        List of mne.epochs.Epochs or strings with filenames. If input
        is a single string or Epochs object, it is first converted into
        a list.

    picks : ndarray of int, optional
        Indices of channels to pick for processing. If None, all
        channels are used.

    target_picks : ndarray of int, optional
        Indices of channels to pick up target variables. If None,
        targets are extracted from other sources. Defaults to None.

    array_keys : dict, optional
        Dictionary mapping {'X': 'data_matrix', 'y': 'labels'},
        where 'data_matrix' and 'labels' are names of the corresponding
        variables, if the input is paths to .mat or .npz files.
        Defaults to {'X': 'X', 'y': 'y'}

    Returns
    -------
    data, targets: ndarray
        data.shape =  [n_epochs, channels, times]

        targets.shape =  [n_epochs, y_shape]

    """
    if isinstance(inp, mne.epochs.BaseEpochs):
        print('processing epochs')
        if isinstance(picks, dict):
            picks = mne.pick_types(inp.info, include=picks)
        inp.load_data()
        data = inp.get_data()
        events = inp.events[:, 2]
        if isinstance(picks, dict):
            print("Converting picks")
            picks = mne.pick_types(inp.info, picks)

    elif isinstance(inp, tuple) and len(inp) == 2:
        print('importing from tuple')
        data, events = inp

    elif isinstance(inp, str):
        # TODO: ADD CASE FOR RAW FILE
        fname = inp
        if fname[-3:] == 'fif':
            epochs = mne.epochs.read_epochs(fname, preload=True,
                                          verbose='CRITICAL')
            print(np.unique(epochs.events[:, 2]))
            events = epochs.events[:, 2]
            data = epochs.get_data()

        else:
            if fname[-3:] == 'mat':
                datafile = sio.loadmat(fname)

            if fname[-3:] == 'npz':
                print('Importing from npz')
                datafile = np.load(fname)

            data = datafile[array_keys['X']]
            if np.any(target_picks):
                events = data[:, target_picks, :]
                data = np.delete(data, target_picks, axis=1)
                print('Extracting target variables from target_picks')
            else:
                events = datafile[array_keys['y']]
#                if (events.shape[0] != data.shape[0]):
#                    if (events.shape[0] == 1):
#                        events = np.squeeze(events, axis=0)
#                    else:
#                        raise ValueError("Target array misaligned.")
                print('Extracting target variables from {}'.format(
                        array_keys['y']))

    data = data.astype(np.float32)

    # TODO: make sure that X is 3d here
    while data.ndim < 3:
        #(x, ) -> (1, 1, x)
        #(x, y) -> (1, x, y)
        data = np.expand_dims(data, 0)

    if isinstance(picks, (np.ndarray, list, tuple)):
        picks = np.asarray(picks)
        if np.any(data.shape[1] <= picks):
            raise ValueError("Invalid picks {} for n_channels {} ".format(
                    max(len(picks), max(picks)), data.shape[1]))
        data = data[:, picks, :]

    if transpose:
        assert isinstance(transpose, (list, tuple)), "Transpose should be list or tuple of str."
        if 'X' in transpose:
            data = np.swapaxes(data, -1, -2)
        if 'y' in transpose:
            if events.ndim >= 2:
                events = np.swapaxes(events, -1, -2)
            else:
                warnings.warn('Targets cannot be transposed.', UserWarning)

    print('input shapes: X-', data.shape, 'targets-', events.shape)
    assert data.ndim == 3, "Import data panic: output.ndim != 3"
    return data, events


def produce_labels(y, return_stats=True):
    """Produce labels array from e.g. event (unordered) trigger codes.

    Parameters
    ----------
    y : ndarray, shape (n_epochs,)
        Array of trigger codes.

    return_stats : bool
        Whether to return optional outputs.

    Returns
    -------
    inv : ndarray, shape (n_epochs)
        Ordered class labels.

    total_counts : int, optional
        Total count of events.

    class_proportions : dict, optional
        {new_class: proportion of new_class1 in the dataset}.

    orig_classes : dict, optional
        Mapping {new_class:old_class}.
    """
    classes, inds, inv, counts = np.unique(y,
                                           return_index=True,
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



def produce_tfrecords(inputs, savepath, out_name, fs=0,
                      input_type='trials', target_type='float',
                      array_keys={'X': 'X', 'y': 'y'}, val_size=0.2,
                      scale=False, scale_interval=None, crop_baseline=False,
                      bp_filter=False, decimate=False, combine_events=None,
                      segment=False, aug_stride=None, seq_length=None,
                      picks=None, transpose=False, target_picks=None,
                      overwrite=True, savebatch=1, test_set=False,
                      transform_targets=False):

    r"""
    Produce TFRecord files from input, apply (optional) preprocessing.

    Calling this function will convert the input data into TFRecords
    format that is used to effiently store and run Tensorflow models on
    the data.


    Parameters
    ----------
    inputs : mne.Epochs, list of str, tuple of ndarrays
        Input data.

    savepath : str
        A path where the output TFRecord and corresponding metadata
        files will be stored.

    out_name : str
        Filename prefix for the output files.

    fs : float, optional
         Sampling frequency, required only if inputs are not mne.Epochs

    input_type : str {'trials', 'continuous', 'seq'}
        Type of input data.

    target_type : str {'int', 'float'}
        Type of target variable.
        'int' - for classification, 'float' for regression problems.

    array_keys : dict, optional
        Dictionary mapping {'X':'data_matrix','y':'labels'},
        where 'data_matrix' and 'labels' are names of the
        corresponding variables if the input is paths to .mat or .npz
        files. Defaults to {'X':'X', 'y':'y'}

    val_size : float, optional
        Proportion of the data to use as a validation set. Only used if
        shuffle_split = True. Defaults to 0.2.

    scale : bool, optional
        Whether to perform scaling to baseline. Defaults to False.

    scale_interval : NoneType, tuple of ints or floats,  optimal
        Baseline definition. If None (default) scaling is
        performed based on all timepoints of the epoch.
        If tuple, then baseline is data[tuple[0] : tuple[1]].
        Only used if scale == True.

    crop_baseline : bool, optional
        Whether to crop baseline specified by 'scale_interval'
        after scaling. Defaults to False.

    bp_filter : bool, tuple, optional
        Band pass filter. Tuple of int or NoneType.

    decimate : False, int, optional
        Whether to decimate the input data. Defaults to False.

    combine_events : dict, optional
        Dictionary for combining or otherwise manipulating lables.
        Should contain mapping {old_label: new_label}. If provided and
        some old_labels are not specified in keys, the corresponding
        epochs are discarded.

    segment : bool, int, optional
        Whether to spit the data into smaller segments of specified
        length.

    augment : bool, optional
        Whether to apply sliding window augmentation.

    aug_stride : int, optional
        Stride of sliding window augmentation.

    picks : ndarray of int, optional
        Array of channel indices to use in decoding.

    target_picks : ndarray, optional
        Array of channel indices used to extract target variable.

    transpose : bool, tuple, optional
        ('X', 'y') swaps last two dimensions of both data and targets
        during import.
        ('X'), does the same for data only. Default is False.

    transform_targets : callable, optional
        custom function transforming target variables

    seq_length : int, optional
        Length of segment sequence.

    overwrite : bool, optional
        Whether to overwrite the metafile if it already exists at the
        specified path.

    savebatch : int
        Number of input files to be stored in each output TFRecord
        file. Defaults to 1.

    test_set : str {'holdout', 'loso', None}, optional
        Defines if a separate holdout test set is required.
        'holdout' saves 50% of the validation set
        'loso' saves the whole dataset in original order for
        leave-one-subject-out cross-validation.
        None does not leave a separate test set. Defaults to None.

    Returns
    -------
    meta : dict
        Metadata associated with the processed dataset. Contains all
        the information about the dataset required for further
        processing with mneflow.
        Whenever the function is called the copy of metadata is also
        saved to savepath/meta.pkl so it can be restored at any time.


    Notes
    -----
    Pre-processing functions are implemented mostly for for convenience
    when working with array inputs. When working with mne.epochs the
    use of the corresponding mne functions is preferred.

    Examples
    --------
    >>> meta = mneflow.produce_tfrecords(input_paths, \**import_opts)

    """
    assert input_type in ['trials', 'seq', 'continuous'], "Unknown input type."
    assert target_type in ['int', 'float', 'signal'], "Unknown target type."
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    if overwrite or not os.path.exists(savepath+out_name+'_meta.pkl'):

        meta = dict(train_paths=[], val_paths=[], test_paths=[],
                    data_id=out_name, train_size=0, val_size=0, test_size=0,
                    savepath=savepath, target_type=target_type,
                    input_type=input_type)
        jj = 0

        meta['fs'] = fs
        if not isinstance(inputs, list):
            inputs = [inputs]
        for inp in inputs:

            data, events = import_data(inp, picks=picks, array_keys=array_keys,
                                       target_picks=target_picks,
                                       transpose=transpose)
            if target_type == 'int':
                # Specific to classification
                if combine_events:
                    events, keep_ind = _combine_labels(events, combine_events)

                    # TODO!  suggest these move inside _combine_labels
                    data = data[keep_ind, ...]
                    events = events[keep_ind]

                events, n_ev, meta['class_ratio'], _ = produce_labels(events)

                events = _onehot(events)

            #elif target_type == 'signal':
            #    print(data.shape, events.shape)

            x_train, y_train, x_val, y_val = preprocess(
                    data, events, input_type=input_type, scale=scale, fs=fs,
                    val_size=val_size, scale_interval=scale_interval,
                    segment=segment, aug_stride=aug_stride,
                    crop_baseline=crop_baseline, decimate=decimate,
                    bp_filter=bp_filter, seq_length=seq_length,
                    transform_targets=transform_targets)

            if test_set == 'holdout':
                x_val, y_val, x_test, y_test = _split_sets(x_val, y_val,
                                                           val=.5)
                meta['test_size'] += x_test.shape[0]

            meta['y_shape'] = y_train[0].shape
            _n, meta['n_seq'], meta['n_t'], meta['n_ch'] = x_train.shape
            meta['train_size'] += _n
            meta['val_size'] += x_val.shape[0]

            print('Prepocessed sample shape:', x_train[0].shape)
            print('Target shape actual/metadata: ',
                  y_train[0].shape, meta['y_shape'])

            print('Saving TFRecord# {}'.format(jj))

            meta['val_size'] += len(y_val)
            meta['train_paths'].append(''.join([savepath, out_name,
                                                '_train_', str(jj),
                                                '.tfrecord']))
            _write_tfrecords(x_train, y_train, meta['train_paths'][-1],
                             target_type=target_type)

            meta['val_paths'].append(''.join([savepath, out_name,
                                              '_val_', str(jj),
                                              '.tfrecord']))
            _write_tfrecords(x_val, y_val, meta['val_paths'][-1],
                             target_type=target_type)

            if test_set == 'loso':
                meta['test_size'] = len(y_val) + len(y_train)
                meta['test_paths'].append(''.join([savepath, out_name,
                                                   '_test_', str(jj),
                                                   '.tfrecord']))
                _write_tfrecords(np.concatenate([x_train, x_val], axis=0),
                                 np.concatenate([y_train, y_val], axis=0),
                                 meta['test_paths'][-1],
                                 target_type=target_type)

            elif test_set == 'holdout':
                meta['test_paths'].append(''.join([savepath, out_name,
                                                   '_test_', str(jj),
                                                   '.tfrecord']))
                _write_tfrecords(x_test, y_test, meta['test_paths'][-1],
                                 target_type=target_type)
            jj += 1
        with open(savepath+out_name+'_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

    elif os.path.exists(savepath+out_name+'_meta.pkl'):
        print('Metadata file found, restoring')
        meta = _load_meta(savepath, data_id=out_name)
    return meta


def _combine_labels(labels, new_mapping):
    """Combine event labels.

    Parameters
    ----------
    labels : ndarray
        Label vector

    combine_dict : dict
        Mapping {new_label1: [old_label1, old_label2], ...}

    Returns
    -------
    new_labels : ndarray
        Updated label vector.

    keep_ind : ndarray
        Label indices.
    """
    assert isinstance(new_mapping, dict), "Invalid label mapping."
    # Find all possible label values
    tmp = []
    for k, j in new_mapping.items():
        tmp.append(k)
        if not isinstance(j, (list, tuple)):
            # for simplicity, force all old_labels to be lists
            new_mapping[k] = [j]
        tmp.extend(new_mapping[k])

    # pick the exlusion value
    inv = np.min(tmp) - 1
    new_labels = inv*np.ones(len(labels), int)

    for new_label, old_label in new_mapping.items():
        # print(old_label, new_label)
        ind = [ii for ii, v in enumerate(labels) if v in old_label]
        new_labels[ind] = int(new_label)
    keep_ind = np.where(new_labels != inv)[0]
    return new_labels, keep_ind


def _segment(data, segment_length=200, seq_length=None, stride=None,
             input_type='trials'):
    """Split the data into fixed-length segments.

    Parameters
    ----------
    data : ndarray
        Data array of shape (n_epochs, n_channels, n_times)

    labels : ndarray
        Array of labels (n_epochs, y_shape)

    seq_length: int or None
        Length of segment sequence.

    segment_length : int or False
        Length of segment into which to split the data in time samples.

    Returns
    -------
    data : list of ndarrays
        Data array of shape
        [x, seq_length, n_channels, segment_length]
        where x = (n_epochs//seq_length)*(n_times - segment_length + 1)//stride
        """
    x_out = []
    if input_type == 'trials':
        seq_length = 1

    if not stride:
            stride = segment_length

    #print(len(data), data[0].shape)
    for jj, xx in enumerate(data):
        #print('xx :', xx.shape)
        n_ch, n_t = xx.shape
        last_segment_start = n_t - segment_length
        #print('last start:', last_segment_start)

        #print("stride:", stride)
        starts = np.arange(0, last_segment_start+1, stride)

        segments = [xx[..., s:s+segment_length] for s in starts]
        #print("n_segm:", len(segments))
        if input_type == 'seq':
            if not seq_length:
                seq_length = len(segments)
            seq_bins = np.arange(seq_length, len(segments)+1, seq_length)
            segments = np.split(segments, seq_bins, axis=0)[:-1]
            x_new = np.array(segments)
        else:
            x_new = np.stack(segments, axis=0)
            x_new = np.expand_dims(x_new, 1)
            #print("x_new:", x_new.shape)
#        if jj == len(data) - 1:
#            print("n_segm:", seq_length)
#            print("x_new:", x_new.shape)
        x_out.append(x_new)
    #print(len(x_out))
    if len(x_out) > 1:
        X = np.concatenate(x_out)
    else:
        X = x_out[0]
    print("X:", X.shape)
    return X


def cont_split_indices(X, test_size=.1, test_segments=5):
    """X - 3d data array."""
    raw_len = X.shape[-1]
    test_samples = int(test_size*raw_len//test_segments)
    interval = raw_len//(test_segments+1)
    data_intervals = np.arange(test_samples, raw_len-test_samples, interval)

    test_start = [ds + np.random.randint(interval - test_samples)
                  for ds in data_intervals]

    test_indices = [(t_strt, t_strt+test_samples)
                    for t_strt in test_start[:-1]]
    #print("test_indices:", test_indices)
    return test_indices


def partition(data, test_indices):
    """Partition continuous data according to ranges defined by `test_indices`.

    Parameters
    ----------
    data : list of ndarray
        Data array to be partitioned.

    test_indices : list
        Contains pairs of values [start, end], indicating where the data
        will be partitioned.

    Returns
    -------
    x_train, x_test: lists of ndarrays
        The data partitioned into two sets.

    Raises
    ------
        ValueError: If the shape of`test_indices` is incorrect.

        AttributeError: If `test_indices` is empty.
    """
    if any(test_indices):
        if np.ndim(test_indices) == 1 and np.max(test_indices) < data.shape[0]:
            x_out = [data[test_indices, ...],
                     np.delete(data, test_indices, axis=0)][::-1]
        elif np.ndim(test_indices) == 2:
            bins = sorted(np.ravel(test_indices))
            # data is a 3d array
            data = np.squeeze(data)
            x_out = np.split(data, bins, axis=-1)
        else:
            raise ValueError('Could not split the data, check test_indices!')

        #x_out = [xt - np.median(xt, axis=-1, keepdims=True) for xt in x_out]
        x_train = x_out[0::2]
        x_test = x_out[1::2]

        #print([xt.shape for xt in x_train])
        #print([xt.shape for xt in x_test])
        return x_train, x_test
    else:
        raise AttributeError('No test indices provided')


def preprocess(data, events, input_type='trials', val_size=.1, scale=False,
               fs=None, scale_interval=None, crop_baseline=False,
               decimate=False, bp_filter=False, picks=None, segment=False,
               aug_stride=None, seq_length=None, transform_targets=None):
    """Preprocess input data.

    Parameters
    ----------
    scale : bool, optional
        Whether to perform scaling to baseline. Defaults to False.

    scale_interval : NoneType, tuple of ints or floats,  optional
        Baseline definition. If None (default) scaling is
        performed based on all timepoints of the epoch.
        If tuple, than baseline is data[tuple[0] : tuple[1]].
        Only used if scale == True.

    crop_baseline : bool, optional
        Whether to crop baseline specified by 'scale_interval'
        after scaling (defaults to False).


    Returns
    -------
    x_train, x_val: ndarrays
        Data arrays of dimensions [n_epochs, n_seq, n_ch, n_t]

    y_train, y_val : ndarrays
        Label arrays of dimensions [n_epochs, n_seq, n_targets]
    """
    print("Preprocessing:")
    if (data.ndim != 3) or (events.ndim < 2):
        warnings.warn('Input misshaped, using import_data.', UserWarning)
        data, events = import_data((data, events))

    if bp_filter:
        #assert len(bp_filter) == 2, "Invalid bp_filter values."
        print('Filtering')
        data = data.astype(np.float64)
        data = mne.filter.filter_data(data, fs, l_freq=bp_filter[0],
                                   h_freq=bp_filter[1],
                                   method='iir', verbose=False)

    if isinstance(picks, np.ndarray):
        data = data[:, picks, :]

    if decimate:
        print("Decimating")
        data = data[..., ::decimate]

    if scale:
        print('Scaling')
        data = scale_to_baseline(data, baseline=scale_interval,
                                 crop_baseline=crop_baseline)

    if input_type in ['continuous']:
#        # Placeholder for future segmentation. Not unit_tested
        test_inds = cont_split_indices(data, test_size=val_size,
                                       test_segments=5)
        x_train, x_val = partition(data, test_inds)
        y_train, y_val = partition(events, test_inds)
        #print("partition:", type(y_train), len(y_train), y_train[0].shape)
    else:
        # TODO (Gabi): Leaving this in as a reminder for the BCI dataset
        # if data.shape[0] == 1:
        #    x_train, y_train, x_val, y_val = _split_sets(data.T, events.T,
        #                                                 val=val_size)
        #   x_train, y_train, x_val, y_val = [ii.T for ii in [x_train, y_train,
        #                                                     x_val, y_val]]
        # else:
        print("Splitting sets")
        x_train, y_train, x_val, y_val = _split_sets(data, events,
                                                     val=val_size)

    if segment:
        #segment each fold separately
        print("Segmenting X")

        x_train = _segment(x_train, segment_length=segment, stride=aug_stride,
                           input_type=input_type, seq_length=seq_length)

        x_val = _segment(x_val, segment_length=segment, stride=aug_stride,
                         input_type=input_type, seq_length=seq_length)

#        if isinstance(y_train, list) or np.ndim(y_train) > 2:
#            y_train = _segment(y_train, segment_length=segment,
#                               stride=aug_stride, input_type=input_type,
#                               seq_length=seq_length)
#            y_val = _segment(y_val, segment_length=segment,
#                             stride=aug_stride, input_type=input_type,
#                             seq_length=seq_length)
        #TODO: add process_y
        if callable(transform_targets):
            print("Transforming targets")
            #print(type(y_train), y_train[0].shape)
            y_train = transform_targets(y_train)
            y_val = transform_targets(y_val)
        else:
            print("Replicating labels for segmented data")
            #repeat label for all subsegments
            y_train = np.repeat(y_train, x_train.shape[0]//y_train.shape[0],
                                axis=0)
            y_val = np.repeat(y_val, x_val.shape[0]//y_val.shape[0], axis=0)
    else:

        x_train = np.expand_dims(x_train, 1)
        x_val = np.expand_dims(x_val, 1)

        if callable(transform_targets):
            print("Transforming targets")
            # print(type(y_train), y_train[0].shape)
            y_train = transform_targets(y_train, decimate=decimate, )
            y_val = transform_targets(y_val)

    x_train = np.swapaxes(x_train, -2, -1)
    x_val = np.swapaxes(x_val, -2, -1)


    print('train preprocessed:', x_train.shape, y_train.shape)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]
    #    print('val preprocessed:', x_val.shape, y_val.shape)

    return x_train, y_train, x_val, y_val

def _process_labels(y, scale=False, decimate=False, normalize=False,
                   transpose=False, transform=False, segment=False):

    #    """Preprocess target variables."""
    y_out = y[:, 0, -50:].mean(-1, keepdims=True)
    #y_out = np.squeeze(np.concatenate(y_out))
    if np.ndim(y_out) == 1:
        y_out = np.expand_dims(y_out, -1)
    print("_process_labels out:", y_out.shape)
    return y_out