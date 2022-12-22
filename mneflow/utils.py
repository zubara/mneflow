# -*- coding: utf-8 -*-
"""
Specifies utility functions.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import os
import pickle
import warnings
import numpy as np
import tensorflow as tf
import scipy.io as sio
import mne


def _onehot(y, n_classes=False):
    """
    Transforms n-by-1 vector of class labels into n-by-n_classes array of
    one-hot encoded labels

    Parameters
    ----------
    y : array of ints
        Array of class labels

    n_classes : int
        Number of classes. If set to False (default) n_classes is set to number of
        unique labels in y


    Returns
    -------
    y_onehot : array
        array of onehot encoded labels

    """
    if not n_classes:
        """Create one-hot encoded labels."""
        n_classes = len(set(y))
    out = np.zeros((len(y), n_classes))
    for i, ii in enumerate(y):
        out[i][ii] += 1
    y_onehot = out.astype(int)
    return y_onehot


def load_meta(fname, data_id=''):
    # TODO: expand functionality?
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




def scale_to_baseline(X, baseline=None, crop_baseline=False):
    """Perform global scaling based on a specified baseline.

    Subtracts the mean of each channel and divides by the standard deviation of
    all channels during the specified baseline interval.

    Parameters
    ----------
    X : ndarray
        Data array with dimensions [n_epochs, n_channels, time].

    baseline : tuple of int, None
        Baseline definition (in samples). If baseline is set to None (default)
        the whole epoch is used for scaling.

    crop_baseline : bool
        Whether to crop the baseline after scaling is applied. Only used if
        baseline is specified.
    Returns
    -------
    X : ndarray
        Scaled data array.

    """
    #X = X_.copy()

    if baseline is None:
        print("No baseline interval specified, sacling based on the whole epoch")
        interval = np.arange(X.shape[-1])
    elif isinstance(baseline, tuple):
        print("Scaling to interval {:.1f} - {:.1f}".format(*baseline))
        interval = np.arange(baseline[0], baseline[1])
    X0m = X[..., interval].mean(axis=2, keepdims=True)
    X0sd = X[..., interval].std(axis=(1,2), keepdims=True)

    X -= X0m
    X /= X0sd
    if crop_baseline and baseline is not None:
        X = np.delete(X, interval, axis=-1)
    #print("Scaling Done")
    return X


def _make_example(X, y, n, target_type='int'):
    """Construct a serializable example proto object from data and
    target pairs."""

    feature = {}
    feature['X'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=X.flatten()))
    feature['n'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=n.flatten()))

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


def _write_tfrecords(X_, y_, n_, output_file, target_type='int'):
    """Serialize and write datasets in TFRecords format.

    Parameters
    ----------
    X_ : list of ndarrays
        (Preprocessed) data matrix.
        len = `n_epochs`, shape = `(squence_length, n_timepoints, n_channels)`

    y_ : list of ndarrays
        Class labels.
        len =  `n_epochs`, shape = `y_shape`

    n_ : int
        nubmer of training examples

    output_file : str
        Name of the TFRecords file.
    """
    writer = tf.io.TFRecordWriter(output_file)

    for X, y, n in zip(X_, y_, n_):
        # print(len(X_), len(y_), X.shape, y.shape)
        X = X.astype(np.float32)
        n = n.astype(np.int64)
        # Feature contains a map of string to feature proto objects
        example = _make_example(X, y, n, target_type=target_type)
        # Serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to the disk
        writer.write(serialized)
    writer.close()


def _split_indices(X, y, n_folds=5):
    # TODO: check if indices are permuted
    """Generate indices for n-fold cross-validation"""
    n = X.shape[0]
    print('n:', n)
    #original_indices = np.arange(n)
    shuffle = np.random.permutation(n)
    subset_proportion = 1./float(n_folds)
    fold_size = int(subset_proportion*n)
    folds = [shuffle[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    return folds


def _split_sets(X, y, folds, ind=-1, sample_counter=0):
    """Split the data returning a single fold specified by ind as a holdout set
        and the rest of the data as training/validation sets.

    Parameters
    ----------
    X : ndarray
        (Preprocessed) data matrix.
        shape (n_epochs, ...)

    y : ndarray
        Class labels.
        shape (n_epochs, ...)

    folds : list of arrays
        fold indices

    ind : index of the selected fold, defaults to -1

    Returns
    -------
    X_train, y_train, X_test, y_test : ndarray
        Pairs of data / targets split in Training and Validation sets.

    test_fold : np.array


    """

    fold = folds.pop(ind) - sample_counter
    X_test = X[fold, ...]
    y_test = y[fold, ...]
    X_train = np.delete(X, fold, axis=0)
    y_train = np.delete(y, fold, axis=0)
    test_fold = fold + sample_counter
    # return X_train, np.squeeze(y_train), X_val, np.squeeze(y_val)
    return X_train, y_train, X_test, y_test, test_fold


def import_data(inp, picks=None, array_keys={'X': 'X', 'y': 'y'}):
    """Import epoch data into `X, y` data/target pairs.

    Parameters
    ----------
    inp : list, mne.epochs.Epochs, str
        List of mne.epochs.Epochs or strings with filenames.
        If input is a single string or Epochs object, it is first converted
        into a list.

    picks : ndarray of int, optional
        Indices of channels to pick for processing. If None, all
        channels are used.

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
        # if isinstance(picks, dict):
        #     picks = mne.pick_types(inp.info, include=picks)
        inp.load_data()
        data = inp.get_data()
        events = inp.events[:, 2]
        if isinstance(picks, dict):
            print("Converting picks")
            picks = mne.pick_types(inp.info, **picks)

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
            if isinstance(picks, dict):
                print("Converting picks")
                picks = mne.pick_types(epochs.info, **picks)

        else:
            if fname[-3:] == 'mat':
                datafile = sio.loadmat(fname)

            if fname[-3:] == 'npz':
                print('Importing from npz')
                datafile = np.load(fname)

            data = datafile[array_keys['X']]
            events = datafile[array_keys['y']]
            print('Extracting target variables from {}'
                  .format(array_keys['y']))
    else:
        print("Dataset not found")
        return None, None

    data = data.astype(np.float32)

    # Make sure that X is 3d here
    while data.ndim < 3:
        # (x, ) -> (1, 1, x)
        # (x, y) -> (1, x, y)
        data = np.expand_dims(data, 0)

    if isinstance(picks, (np.ndarray, list, tuple)):
        picks = np.asarray(picks)
        if np.any(data.shape[1] <= picks):
            raise ValueError("Invalid picks {} for n_channels {} ".format(
                    max(len(picks), max(picks)), data.shape[1]))
        data = data[:, picks, :]

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
        {new_class: proportion of new_class in the dataset}.

    orig_classes : dict, optional
        Mapping {new_class_label: old_class_label}.
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


def produce_tfrecords(inputs, savepath, out_name, fs=1.,
                      input_type='trials',
                      target_type='int',
                      array_keys={'X': 'X', 'y': 'y'},
                      n_folds=5,
                      scale=False,
                      scale_interval=None,
                      crop_baseline=False,
                      segment=False,
                      aug_stride=None,
                      seq_length=None,
                      picks=None,
                      overwrite=False,
                      test_set=False,
                      bp_filter=False,
                      decimate=False,
                      combine_events=None,
                      transform_targets=False,
                      scale_y=False,
                      save_as_numpy=False):

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

        'trials' - treats each of n inputs as an iid sample, produces dataset
        with dimensions (n, 1, t, ch)

        'seq' - treats each of n inputs as a seqence of shorter segments,
                produces dataset with dimensions
                (n, seq_length, segment, ch)

        'continuous' - treats inputs as a single continuous sequence,
                        produces dataset with dimensions
                        (n*(t-segment)//aug_stride, 1, segment, ch)

    target_type : str {'int', 'float'}
        Type of target variable.

        'int' - for classification,
        'float' - for regression problems.
        'signal' - regression or classification a continuous (possbily
                   multichannel) data. Requires "transform_targets" function
                   to apply to targets

    n_folds : int, optional
        Number of folds to split the data for training/validation/testing.
        One fold of the n_folds is used as a validation set.
        If test_set == 'holdout' generates one extra fold
        used as test set. Defaults to 5

    test_set : str {'holdout', 'loso', None}, optional
        Defines if a separate holdout test set is required.
        'holdout' saves 50% of the validation set
        'loso' saves the whole dataset in original order for
        leave-one-subject-out cross-validation.
        None does not leave a separate test set. Defaults to None.


    segment : bool, int, optional
        If specified, splits the data into smaller segments of specified
        number of time points. Defaults to False

    aug_stride : int, optional
        Sliding window agumentation stride parameter.
        If specified, sets the stride (in time points) for 'segment'
        allowing to extract overalapping segments. Has to be <= segment.
        Only applied within each fold to prevent data leakeage. Only applied
        if 'segment' is not False. If None, then it is set equal to length of
        the 'segment' returning non-overlapping segments.
        Defaults to None.


    scale : bool, optional
        Whether to perform scaling to baseline. Defaults to False.

    scale_interval : NoneType, tuple of ints,  optional

        Baseline definition. If None (default) scaling is
        performed based on all timepoints of the epoch.
        If tuple, then baseline is data[tuple[0] : tuple[1]].
        Only used if scale == True.

    crop_baseline : bool, optional
        Whether to crop baseline specified by 'scale_interval'
        after scaling. Defaults to False.

    array_keys : dict, optional
        Dictionary mapping {'X':'data_matrix','y':'labels'},
        where 'data_matrix' and 'labels' are names of the
        corresponding variables if the input is paths to .mat or .npz
        files. Defaults to {'X':'X', 'y':'y'}

    bp_filter : bool, tuple, optional
        Band pass filter. Tuple of int or NoneType.

    decimate : False, int, optional
        Whether to decimate the input data. Defaults to False.

    combine_events : dict, optional
        Dictionary for combining or otherwise manipulating lables.
        Should contain mapping {old_label: new_label}. If provided and
        some old_labels are not specified in keys, the corresponding
        epochs are discarded.

    picks : ndarray of int, optional
        Array of channel indices to use in decoding.

    transform_targets : callable, optional
        custom function used to transform target variables

    seq_length : int, optional
        Length of segment sequence.

    overwrite : bool, optional
        Whether to overwrite the metafile if it already exists at the
        specified path.

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
                    folds=[], test_fold=[],
                    data_id=out_name, train_size=0, val_size=0, test_size=0,
                    savepath=savepath, target_type=target_type,
                    input_type=input_type)
        jj = 0
        if test_set == 'holdout':
            n_folds += 1

        meta['fs'] = fs

        if not isinstance(inputs, list):
            inputs = [inputs]

        for inp in inputs:

            data, events = import_data(inp, picks=picks, array_keys=array_keys)

            if np.any(data) == None:
                return

            else:

                #if target_type == 'int':
                    # Specific to classification
#                    if combine_events:
#                        events, keep_ind = _combine_labels(events, combine_events)
#
#                        # TODO!  suggest these move inside _combine_labels
#                        data = data[keep_ind, ...]
#                        events = events[keep_ind]
#

                # Check label dimensions again
                if input_type == 'continuous':
                    # if input is a continuous signal ensure that target
                    # variable has shape (n_epochs, channels, time)
                    # TODO: replace with "target type?"
                    while events.ndim < 3:
                        events = np.expand_dims(events, 0)
                else:
                    # if input is trials, ensure that target variable has shape
                    # (n_trials, y_shape)
                    if events.ndim < 2:
                        events = np.expand_dims(events, -1)

                if input_type == 'trials':
                    segment_y = False
                else:
                    segment_y = True

                print('Input shapes: X (n, ch, t) : ', data.shape,
                      'y (n, [signal_channels], y_shape) : ', events.shape,
                      '\n',
                      'input_type : ', input_type,
                      'target_type : ', target_type,
                      'segment_y : ', segment_y)

#                if (data.ndim != 3):
#                    warnings.warn('Input misshaped, using import_data.', UserWarning)
#                    return
                #Preprocess data and segment labels if needed
                # TODO define segment_y
                X, Y, folds = preprocess(
                        data, events,
                        sample_counter=meta['train_size'],
                        input_type=input_type,
                        n_folds=n_folds,
                        scale=scale,
                        scale_interval=scale_interval,
                        crop_baseline=crop_baseline,
                        segment=segment, aug_stride=aug_stride,
                        seq_length=seq_length,
                        segment_y=segment_y)

                Y = preprocess_targets(Y, scale_y=scale_y,
                                       transform_targets=transform_targets)

                if target_type == 'int':
                    Y, n_ev, meta['class_ratio'], meta['orig_classes'] = produce_labels(Y)
                    Y = _onehot(Y)


                if test_set == 'holdout':
                    X, Y, x_test, y_test, test_fold = _split_sets(X, Y,
                                                                  folds=folds,
                                                                  sample_counter=meta['train_size'])
                    meta['test_size'] += x_test.shape[0]
                    #TODO: remove?
                _n, meta['n_seq'], meta['n_t'], meta['n_ch'] = X.shape

                if input_type == 'seq':
                    meta['y_shape'] = Y[0].shape[1:]
                else:
                    meta['y_shape'] = Y[-1].shape

                n = np.arange(_n) + meta['train_size']

                meta['train_size'] += _n

                if save_as_numpy == True:
                    train_fold = np.concatenate(folds[1:])
                    val_fold = folds[0]
                    np.savez(savepath+out_name,
                             X_train=np.swapaxes(X[train_fold, ...], -2, -1),
                             X_val=np.swapaxes(X[val_fold, ...], -2, -1),
                             #X_test=np.swapaxes(x_test,-2, -1),
                             y_train=Y[train_fold, ...],
                             y_val=Y[val_fold, ...],
                             #y_test=y_test
                             )


                #                                                 np.min(n), np.max(n)))
                meta['val_size'] += len(folds[0])

                print('Prepocessed sample shape:', X[0].shape)
                print('Target shape actual/metadata: ', Y[0].shape, meta['y_shape'])

                print('Saving TFRecord# {}'.format(jj))

                meta['folds'].append(folds)
                meta['train_paths'].append(''.join([savepath, out_name,
                                                    '_train_', str(jj),
                                                    '.tfrecord']))

                _write_tfrecords(X, Y, n, meta['train_paths'][-1],
                                 target_type=target_type)

                if test_set == 'loso':
                    meta['test_size'] = len(Y)
                    meta['test_paths'].append(''.join([savepath, out_name,
                                                       '_test_', str(jj),
                                                       '.tfrecord']))
                    _write_tfrecords(X, Y, n, meta['test_paths'][-1],
                                     target_type=target_type)

                elif test_set == 'holdout':
                    meta['test_fold'].append(test_fold)

                    meta['test_paths'].append(''.join([savepath, out_name,
                                                       '_test_', str(jj),
                                                       '.tfrecord']))
                    n_test = np.arange(len(test_fold))
                    _write_tfrecords(x_test, y_test, n_test, meta['test_paths'][-1],
                                     target_type=target_type)
                jj += 1
                with open(savepath+out_name+'_meta.pkl', 'wb') as f:
                    pickle.dump(meta, f)

    elif os.path.exists(savepath+out_name+'_meta.pkl'):
        print('Metadata file found, restoring')
        meta = load_meta(savepath, data_id=out_name)
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
    print(labels)
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
    #print(new_labels, keep_ind)
    return new_labels, keep_ind


def _segment(data, segment_length=200,
             seq_length=None,
             stride=None,
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

    stride : int, optional
        If specified, sets the stride (in time points) for 'segment'
        allowing to extract overalapping segments. Has to be <= segment.
        Only applied within each fold to prevent data leakeage. Only applied
        if 'segment' is not False. If None, then it is set equal to length of
        the 'segment' returning non-overlapping segments.
        Defaults to None.

    Returns
    -------
    data : ndarray
        Segmented data array of shape
        (n, [seq_length,] n_channels, segment_length)
        where n = (n_epochs//seq_length)*(n_times - segment_length + 1)//stride
        """
    x_out = []
    if input_type == 'trials':
        seq_length = 1

    if not stride:
        stride = segment_length

    for jj, xx in enumerate(data):

        n_ch, n_t = xx.shape
        last_segment_start = n_t - segment_length

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
#            if not events:
#                x_new = np.expand_dims(x_new, 1)

        x_out.append(x_new)
    if len(x_out) > 1:
        X = np.concatenate(x_out)
    else:
        X = x_out[0]
    print("Segmented as: {}".format(input_type), X.shape)
    return X


def cont_split_indices(data, events, n_folds=5, segments_per_fold=10):
    """
    Parameters
    ----------
    data : ndarray
            3d data array (n, ch, t)

    n_folds : int
             number of folds

    segments_per_fold : int
                        minimum number of different (non-contiguous)
                        data segments in each fold
    Returns
    -------
    data : ndarray
           3d data array (n, ch, t)

    events : nd.array
           labels

    folds : list of ndarrays
            indices for each fold

    """
    raw_len = data.shape[-1]
    # Define minimal duration of a single, non-overlapping data segment
    ind_samples = int(raw_len//(segments_per_fold*n_folds))

    #interval = raw_len//(test_segments+1)
    segments = np.arange(0, raw_len - ind_samples + 1, ind_samples)
    data = np.concatenate([data[:, :, s: s + ind_samples] for s in segments])
    #mod = raw_len - (segments[-1] + ind_samples)
    #data = data[:, :, mod:]
    # Split continous data into non-overlapping segments
    #data = data.reshape([-1, ind_samples, data.shape[-2]])

    #Treat events the same depending on their type
    #case 1: events are signal -> split similarly to the data
    events = np.concatenate([events[:, :, s: s + ind_samples] for s in segments])


    folds = _split_indices(data, events, n_folds=n_folds)
    #test_start = [ds + np.random.randint(interval - test_samples)
    #              for ds in data_intervals]

    #test_indices = [(t_strt, t_strt+test_samples)
    #                for t_strt in test_start[:-1]]
    #print("test_indices:", test_indices)
    return data, events, folds


def preprocess_realtime(data, decimate=False, picks=None,
                        bp_filter=False, fs=None):
    """
    Implements minimal prprocessing for convenitent real-time use.

    Parameters
    ----------
    data : np.array, (n_epochs, n_channels, n_times)
           input data array

    picks : np.array
            indices of channels to pick

    decimate : int
                decimation factor for downsampling

    bp_filter : tuple of ints
                Band-pass filter cutoff frequencies

    fs : int
         sampling frequency. Only used if bp_filter is used

    Returns
    -------
    """
    if bp_filter:
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
    return data


def preprocess(data, events, sample_counter,
               input_type='trials', n_folds=5,
               scale=False, scale_interval=None, crop_baseline=False,
               segment=False, aug_stride=None,
               seq_length=None,
               segment_y=False):
    """Preprocess input data. Applies scaling, segmenting/augmentation,
     and defines the split into training/validation folds.

    Parameters
    ----------
    data : np.array, (n_epochs, n_channels, n_times)
           input data array

    events : np.array
            input array of target variables (n_epochs, ...)

    input_type : str {'trials', 'continuous'}
            See produce_tfrecords.

    n_folds : int
            Number of folds defining the train/validation/test split.

    sample_counter : int
            Number of traning examples in the dataset

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

    segment : bool, int, optional
        If specified, splits the data into smaller segments of specified
        number of time points. Defaults to False

    aug_stride : int, optional
        If specified, sets the stride (in time points) for 'segment'
        allowing to extract overalapping segments. Has to be <= segment.
        Only applied within each fold to prevent data leakeage. Only applied
        if 'segment' is not False. If None, then it is set equal to length of
        the 'segment' returning non-overlapping segments.
        Defaults to None.

    seq_length: int or None
        Length of segment sequence.

    segment_y : bool
        whether to segment target variable in the same way as data. Only used
        if segment != False


    Returns
    -------
    X: np.array
        Data array of dimensions [n_epochs, n_seq, n_t, n_ch]

    Y : np.array
        Label arrays of dimensions [n_epochs, *(y_shape)]

    folds : list of np.arrays

    """
    print("Preprocessing:")

    # TODO: remove scale_y and transform targets?

    if scale:
        data = scale_to_baseline(data, baseline=scale_interval,
                                 crop_baseline=crop_baseline)

    #define folds
    if input_type  == 'continuous':
        data, events, folds = cont_split_indices(data, events,
                                                 n_folds=5,
                                                 segments_per_fold=10)
        print("Continuous events: ", events.shape)

    else:
        folds = _split_indices(data, events, n_folds=n_folds)

    print("Splitting into: {} folds x {}".format(len(folds), len(folds[0])))

    if segment:
        print("Segmenting")
        X = []
        Y = []
        segmented_folds = []
        jj = 0
        for fold in folds:
            #print(data[fold, ...].shape)
            x = _segment(data[fold, ...], segment_length=segment,
                         stride=aug_stride, input_type=input_type,
                         seq_length=seq_length)

            nsegments = x.shape[0]

            # if segment_y -> segment, else-> replicate
            if segment_y:
                y = _segment(events[fold, ...], segment_length=segment,
                             stride=aug_stride, input_type=input_type,
                             seq_length=seq_length)
            else:
                print("Replicating labels for segmented data")
                y = np.repeat(events[fold, ...], nsegments//len(fold), axis=0)

            if x.ndim == 3:
                x = np.expand_dims(x, 1)
            X.append(x)
            Y.append(y)
            segmented_folds.append(np.arange(jj, jj + nsegments) + sample_counter)
            jj += nsegments
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        folds = segmented_folds
    else:
        # If not segmented add a singleton "n_seq" dminesion to X
        if X.ndim == 3:
            X = np.expand_dims(data, 1)

        Y = events
        folds = [f + sample_counter for f in folds]
    # Finally cast X into shape [n_epochs, n_seq, n_times, n_channels]
    X = np.swapaxes(X, -2, -1)

    print('Preprocessed:', X.shape, Y.shape,
          'folds:', len(folds), 'x', len(folds[0]))
    assert X.shape[0] == Y.shape[0], "n_epochs in X ({}) does not match n_epochs in Y ({})".format(X.shape[0], Y.shape[0])

    return X, Y, folds

def preprocess_targets(y, scale_y=False, transform_targets=None):

    if callable(transform_targets):
        y = transform_targets(y)

    if scale_y:
            y -= y.mean(axis=0, keepdims=True)
            y /= y.std(axis=0, keepdims=True)
    print('Preprocessed targets: ', y.shape)

    return y




    #    """Preprocess target variables."""
    # y_out = y[:, 0, -50:].mean(-1, keepdims=True)
    # #y_out = np.squeeze(np.concatenate(y_out))
    # if np.ndim(y_out) == 1:
    #     y_out = np.expand_dims(y_out, -1)
    # print("_process_labels out:", y_out.shape)
    return y_out

def regression_metrics(y_true, y_pred):
    y_shape = y_true.shape[-1]

    cc = np.diag(np.corrcoef(y_true.T, y_pred.T)[:y_shape,-y_shape:])
    r2 =  r2_score(y_true, y_pred)
    cs = cosine_similarity(y_true, y_pred)
    bias = np.mean(y_true, axis=0) - np.mean(y_pred, axis=0)
    #ve = pve(y_true, y_pred)
    return dict(cc=cc, r2=r2, cs=cs, bias=bias)

def cosine_similarity(y_true, y_pred):
    # y_true -= y_true.mean()
    # y_pred -= y_pred.mean()

    return np.dot(y_pred.T, y_true) / (np.sqrt(np.sum(y_pred**2,axis=0)) * np.sqrt(np.sum(y_true**2, axis=0)))

def pve(y_true, y_pred):
    y_true -= y_true.mean(axis=0)
    y_pred -= y_pred.mean(axis=0)
    return np.dot(y_pred.T, y_true) / np.sum(y_pred**2, axis=0)

def r2_score(y_true, y_pred):
    res = np.sum((y_true - y_pred)**2, axis=0)
    tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True))**2, axis=0)
    return 1 - res/tot

