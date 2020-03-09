# -*- coding: utf-8 -*-
"""
Specifies utility functions.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import os
import pickle
from operator import itemgetter

import numpy as np
import tensorflow as tf
import scipy.io as sio

from mne import filter as mnefilt
from mne import epochs as mnepochs  # , pick_types

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
    """Loads a metadata file.

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
    """Perform a leave-one-out cross-validation such that, on each fold
    one input .tfrecord file is used as a validation set.

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

        # TODO! Gabi: reuse of i variable, ensure it doesn't mess up things
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


def _make_example(X, y, input_type='iid', target_type='int'):
    """Construct Example proto object from data / target pairs."""
    # if input_type in ['iid', 'trials']:
    feature = {}
    feature['X'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=X.flatten()))

    if target_type == 'int':
        feature['y'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=y.flatten()))
    elif target_type == 'float':
        y = y.astype(np.float32)
        feature['y'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=y.flatten()))
    else:
        raise ValueError('Invalid target type.')

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # elif input_type == 'seq':
    #    seq_length = X.shape[0]
    #    # print('make_example_shape x', X.shape, seq_length)
    #    X_ = tf.train.FeatureList(feature=[tf.train.Feature(
    #            float_list=tf.train.FloatList(value=s.flatten())) for s in X])
    #    y_ = tf.train.FeatureList(feature=[tf.train.Feature(
    #            float_list=tf.train.FloatList(value=l.flatten())) for l in y])
    #    example = tf.train.SequenceExample(
    #            context={},
    #            feature_lists=tf.train.FeatureLists(
    #                    feature_list={'X': X_, 'y': y_}))
    #    example.context.feature["length"].int64_list.value.append(seq_length)
    # print(token.shape)

    return example


def _write_tfrecords(X_, y_, output_file, input_type='iid', target_type='int'):
    # TODO! Gabi: the parameter description was incorrect, X_ and y_ are
    # lists of arrays. Otherwise zip wouldn't work.
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
        example = _make_example(X, y, input_type=input_type,
                                target_type=target_type)
        # Serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to the disk
        writer.write(serialized)
    writer.close()


def _split_sets(X, y, val=.1):
    """Applies shuffle and splits the shuffled data.

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
    shuffle = np.random.permutation(X.shape[0])
    val_size = int(round(val*X.shape[0]))
    X_val = X[shuffle[:val_size], ...]
    y_val = y[shuffle[:val_size], ...]
    X_train = X[shuffle[val_size:], ...]
    y_train = y[shuffle[val_size:], ...]
    return X_train, np.squeeze(y_train), X_val, np.squeeze(y_val)


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
    if isinstance(inp, mnepochs.BaseEpochs):
        print('processing epochs')
        inp.load_data()
        data = inp.get_data()
        events = inp.events[:, 2]

    elif isinstance(inp, tuple) and len(inp) == 2:
        print('importing from tuple')
        data, events = inp

    elif isinstance(inp, str):
        # TODO: ADD CASE FOR RAW FILE
        fname = inp
        if fname[-3:] == 'fif':
            epochs = mnepochs.read_epochs(fname, preload=True,
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
                print('Extracting target variables from {}'.format(
                        array_keys['y']))

    data = data.astype(np.float32)
    # TODO: make sure that X is 3d here
    while data.ndim < 3:
        data = np.expand_dims(data, 0)
        events = np.expand_dims(events, 0)

    if isinstance(picks, np.ndarray):
        data = data[:, picks, :]

    if transpose:
        if 'X' in transpose:
            data = np.swapaxes(data, -1, -2)
        if 'y' in transpose:
            events = np.swapaxes(events, -1, -2)

    print('input shapes: X-', data.shape, 'targets-', events.shape)
    return data, events



def produce_labels(y, return_stats=True):
    """Produces labels array from e.g. event (unordered) trigger codes.

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


def produce_tfrecords(inputs, savepath, out_name, fs,
                      input_type='trials', target_type='float',
                      array_keys={'X': 'X', 'y': 'y'}, val_size=0.2,
                      scale=False, scale_interval=None,   crop_baseline=False,
                      bp_filter=False, decimate=False, combine_events=None,
                      segment=False, augment=False, aug_stride=50,
                      picks=None, transpose=False,
                      target_picks=None, transform_targets=False,
                      seq_length=None, overwrite=True, savebatch=1,
                      test_set=False,):

    r"""
    Produces TFRecord files from input, applies (optional) preprocessing.

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

    input_type : str {'trials', 'iid', 'seq'}
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

    scale : bool, optinal
        Whether to perform scaling to baseline. Defaults to False.

    scale_interval : NoneType, tuple of ints or floats,  optimal
        Baseline definition. If None (default) scaling is
        performed based on all timepoints of the epoch.
        If tuple, then baseline is data[tuple[0] : tuple[1]].
        Only used if scale == True.

    crop_baseline : bool, optinal
        Whether to crop baseline specified by 'scale_interval'
        after scaling. Defaults to False.

    bp_filter : bool, tuple, optinal
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

    transform_targets : bool, optional
        Whether to transform the targets,

    seq_length : int, optional
        Length of segment sequence.

    overwrite : bool, optional
        Whether to overwrite the metafile if it already exists at the
        specified path.

    savebatch : int
        Number of input files per to be stored in the output TFRecord
        file. Deafults to 1.

    test_set : str {'holdout', 'loso', 'none'}, optinal
        Defines if a separate holdout test set is required.
        'holdout' saves 50% of the validation set
        'loso' saves the whole dataset in original order for
        leave-one-subject-out cross-validation.
        'none' does not leave a separate test set. Defaults to 'none'.

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
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    if overwrite or not os.path.exists(savepath+out_name+'_meta.pkl'):

        meta = dict(train_paths=[], val_paths=[], test_paths=[],
                    data_id=out_name, val_size=0, savepath=savepath,
                    target_type=target_type, input_type=input_type)
        jj = 0
        # i = 0

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
                    data = data[keep_ind, ...]
                    events = events[keep_ind]

                events, n_ev, meta['class_ratio'], _ = produce_labels(events)
                events = _onehot(events)

            x_train,  y_train, x_val, y_val = preprocess(data, events,
                                                         input_type=input_type,
                                                         scale=True, fs=fs,
                                                         val_size=val_size,
                                                         scale_interval=scale_interval,
                                                         segment=segment,
                                                         augment=augment,
                                                         aug_stride=aug_stride,
                                                         crop_baseline=crop_baseline,
                                                         decimate=decimate,
                                                         bp_filter=bp_filter,
                                                         seq_length=seq_length)

            if test_set == 'holdout':
                x_val, y_val, x_test, y_test = _split_sets(x_val, y_val,
                                                           val=.5)

            if input_type == 'trials':
                meta['y_shape'] = y_train[0].shape
            else:
                meta['y_shape'] = y_train[0].shape[-1]

            if input_type == 'seq':
                meta['n_seq'], meta['n_ch'], meta['n_t'] = x_train[0].shape
                meta['y_shape'] = y_train[0].shape[-1]
            else:
                n_epochs, meta['n_ch'], meta['n_t'] = x_train.shape
                meta['y_shape'] = y_train.shape[-1]

            print('Prepocessed sample shape:', x_train[0].shape)
            print('Target shape actual/metadata: ',
                  y_train[0].shape, meta['y_shape'])
            print('Saving TFRecord# {}'.format(jj))

            meta['val_size'] += len(y_val)
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

            if test_set == 'loso':
                meta['test_paths'].append(''.join([savepath, out_name,
                                                   '_test_', str(jj),
                                                   '.tfrecord']))
                _write_tfrecords(np.concatenate([x_train, x_val], axis=0),
                                 np.concatenate([y_train, y_val], axis=0),
                                 meta['test_paths'][-1],
                                 input_type=input_type,
                                 target_type=target_type)
            elif test_set == 'holdout':
                meta['test_paths'].append(''.join([savepath, out_name,
                                                   '_test_', str(jj),
                                                   '.tfrecord']))
                _write_tfrecords(x_test, y_test, meta['test_paths'][-1],
                                 input_type=input_type,
                                 target_type=target_type)
            jj += 1
        with open(savepath+out_name+'_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

    elif os.path.exists(savepath+out_name+'_meta.pkl'):
        print('Metadata file found, restoring')
        meta = _load_meta(savepath, data_id=out_name)
    return meta


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


def _segment(data, segment_length=200, seq_length=None, augment=False,
             stride=None, input_type='iid'):
    """
    Parameters:
    -----------
    data : list of ndarrays
            data array of shape (n_epochs, n_channels, n_times)

    labels : ndarray
            array of labels (n_epochs,)

    segment_length : int or False
                    length of segment into which to split the data in
                    time samples

    """
    x_out = []
    for jj, xx in enumerate(data):
        n_ch, n_t = xx.shape
        last_segment_start = n_t - segment_length
        if not augment:
            stride = segment_length
        starts = np.arange(0, last_segment_start+1, stride)
        segments = [xx[..., s:s+segment_length] for s in starts]

        if input_type == 'seq':
            if not seq_length:
                seq_length = len(segments)
            seq_bins = np.arange(seq_length, len(segments)+1, seq_length)
            segments = np.split(segments, seq_bins, axis=0)[:-1]
            x_new = np.array(segments)
        else:
            x_new = np.stack(segments, axis=0)
        x_out.append(x_new)

    return np.concatenate(x_out, 0)


def cont_split_indices(X, test_size=.1, test_segments=5):
    """
    X - 3d data array

    """
    raw_len = X.shape[-1]
    test_samples = int(test_size*raw_len//test_segments)
    interval = raw_len//(test_segments+1)
    data_intervals = np.arange(test_samples, raw_len-test_samples, interval)

    test_start = [ds + np.random.randint(interval - test_samples)
                  for ds in data_intervals]

    test_indices = [(t_strt, t_strt + test_samples)
                    for t_strt in test_start[:-1]]
    return test_indices


def partition(data, test_indices):

    if any(test_indices):

        if np.ndim(test_indices) == 1 and np.max(test_indices) < data.shape[0]:

            x_out = [data[test_indices, ...], np.delete(data, test_indices,
                                                        axis=0)][::-1]
        elif np.ndim(test_indices) == 2:
            bins = sorted(np.ravel(test_indices))
            x_out = np.split(data, bins, axis=-1)

        else:
            print('Could not split the data, check test_indices!')
            return

        x_out = [xt - np.median(xt, axis=-1, keepdims=True) for xt in x_out]
        x_train = x_out[0::2]
        x_test = x_out[1::2]

        return x_train, x_test
    else:
        print('No test labels provided')
        return None


def preprocess(data, events, input_type='trials', val_size=.1, scale=False,
               fs=None, scale_interval=None, crop_baseline=False,
               decimate=False, bp_filter=False, picks=None, segment=False,
               augment=False, aug_stride=25, seq_length=None):
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

    if input_type == 'continuous':
        test_inds = cont_split_indices(data, test_size=val_size,
                                       test_segments=5)
        x_train, x_val = partition(data, test_inds)
        y_train, y_val = partition(events, test_inds)

    else:
        x_train, y_train, x_val, y_val = _split_sets(data, events,
                                                     val=val_size)
        if y_train.ndim == 1 and y_val.ndim == 1:
            y_train = np.expand_dims(y_train, -1)
            y_val = np.expand_dims(y_val, -1)
    print('training set: X-', x_train.shape, ' y-', y_train.shape)
    print('validation set: X-', x_val.shape, ' y-', y_val.shape)

    if segment:
        x_train = _segment(x_train, segment_length=segment, augment=augment,
                           stride=aug_stride, input_type=input_type,
                           seq_length=seq_length)

        x_val = _segment(x_val, segment_length=segment, augment=augment,
                         stride=aug_stride, input_type=input_type,
                         seq_length=seq_length)
        if y_train.ndim == 3 and y_val.ndim == 3:
            y_train = _segment(y_train, segment_length=segment,
                               augment=augment,
                               stride=aug_stride, input_type=input_type,
                               seq_length=seq_length)
            y_val = _segment(y_val, segment_length=segment, augment=augment,
                             stride=aug_stride, input_type=input_type,
                             seq_length=seq_length)
        else:
            y_train = np.repeat(y_train, x_train.shape[0]//y_train.shape[0], axis=0)
            y_val = np.repeat(y_val, x_val.shape[0]//y_val.shape[0], axis=0)

        print('train segmented:', x_train.shape, y_train.shape)
        print('val segmented:', x_val.shape, y_val.shape)

    return x_train, y_train, x_val, y_val

# def process_labels(y, scale=False, decimate=False, normalize=False,
#                   transpose=False, transform=False, segment=False):
#    """Preprocess target variables."""
#    if transpose:
#        y = np.swapaxes(y, -2, -1)
#
#    if segment:
#        y, _ = _segment(y, labels=None, segment_length=segment)
#    if decimate:
#        assert y.ndim == 3
#        y = y[..., ::decimate]
#    if normalize:
#        y = scale_to_baseline(y, baseline=None, crop_baseline=False)
#    y = np.mean(y**2, axis=-1)
    # print('y', y.shape)
#    if isinstance(transform, callable):
#        y = transform(y)
#    return y

#def preprocess_continuous_labels(data, targets, scale=True, segment=200,
#                                 augment=False, val_size=0.1, aug_stride=10,
#                                 transform_targets=True, bp_filter=False,
#                                 fs=None, decimate=None, input_type='trials',
#                                 seq_length=None):
#    """
#    Returns
#    -------
#    x_train, x_val - arrays or lists of arrays of dimensions [n_epochs, n_seq, n_ch, n_t]
#    y_train, y_val - arrays or lists of arrays of dimensions [n_epochs, n_seq, n_targets]
#    """
#
#    if transform_targets:
#
#        if input_type == 'seq':
#            y_train = [np.mean(y_tr[..., 0, -aug_stride:], axis=-1, keepdims=True) for y_tr in y_train]
#            y_val = [np.mean(y_v[..., 0, -aug_stride:], axis=-1, keepdims=True) for y_v in y_val]
##            y_train = [np.mean(y_tr[..., -aug_stride:], axis=-1) for y_tr in y_train]
##            y_val = [np.mean(y_v[..., -aug_stride:], axis=-1) for y_v in y_val]
#            print('preproc cont', len(x_train), x_train[0].shape, y_train[0].shape)
#        else:
#            y_train = np.mean(y_train[...,0, -aug_stride:], axis=-1, keepdims=True)
#            y_val = np.mean(y_val[...,0, -aug_stride:], axis=-1, keepdims=True)
##            y_train -= y_median[None,0, :]
##            y_val -= y_median[None,0,  :]
##            y_train /= qrange[None,0, :]
##            y_val /= qrange[None, 0, :]
#            print('preproc cont', x_train.shape, y_train.shape)
#    return x_train, y_train, x_val, y_val