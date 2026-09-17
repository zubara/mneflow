#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines mneflow.Dataset object.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import tensorflow as tf
#TODO: fix batching/epoching with training
#TODO: dataset size form h_params

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
from mneflow.utils import _onehot

class Dataset(object):
    """TFRecords dataset from TFRecords files using the metadata.

    Wraps one or more sets of TFRecords files (produced by
    ``mneflow.utils.produce_tfrecords``) into ``tf.data.Dataset``
    objects that are ready to be consumed by ``mneflow`` models,
    applying any requested subsetting, decimation, cropping, and
    class rebalancing.

    Attributes
    ----------
    h_params : dict
        Metadata dictionary (``meta.data``), updated in-place with
        the effective values of ``channel_subset``, ``class_subset``,
        ``sample_subset``, ``decim``, ``crop``, ``train_batch``,
        ``test_batch``, and ``rebalance_classes``.

    y_shape : tuple
        Shape of the target variable, taken from
        ``h_params['y_shape']`` and possibly updated by
        ``_preprocess`` when a ``class_subset`` is applied.

    train : tf.data.Dataset
        Training fold of the dataset built from
        ``h_params['train_paths']``.

    val : tf.data.Dataset
        Validation fold of the dataset built from
        ``h_params['train_paths']``.

    test : tf.data.Dataset
        Held-out test dataset built from ``h_params['test_paths']``.
        Only set if ``h_params['test_paths']`` is non-empty.

    train_fold : ndarray
        Concatenated example indices assigned to the training folds,
        set by :meth:`_build_dataset`.

    val_fold : ndarray
        Example indices assigned to the validation fold, set by
        :meth:`_build_dataset`.

    train_inds : ndarray
        Concatenated indices into the training folds, set by
        :meth:`_build_dataset` when ``'indices'`` is present in
        ``h_params``.

    val_inds : ndarray
        Indices into the validation fold, set by
        :meth:`_build_dataset` when ``'indices'`` is present in
        ``h_params``.

    training_batch : int
        Effective training mini-batch size, set by
        :meth:`_build_dataset`.

    validation_batch : int
        Effective validation mini-batch size, set by
        :meth:`_build_dataset`.

    training_steps : int
        Number of mini-batches per training epoch, set by
        :meth:`_build_dataset`.

    validation_steps : int
        Number of mini-batches per validation epoch, set by
        :meth:`_build_dataset`.

    test_batch : int
        Effective test mini-batch size, set by :meth:`_build_dataset`
        when ``split=False``.

    test_steps : int
        Number of mini-batches per test epoch, set by
        :meth:`_build_dataset` when ``split=False``.

    sample_subset : ndarray
        Index array used to gather a subset of samples, set by
        :meth:`_build_dataset` when
        ``h_params['sample_subset']`` is not None.

    timepoint_subset : array-like
        Indices of timepoints to keep, set by :meth:`_build_dataset`
        when ``h_params['timepoint_subset']`` is not None.

    """

    def __init__(self, meta, train_batch=50, test_batch=None, split=True,
                 class_subset=None, pick_channels=None, decim=None,
                 sample_subset=None, crop=None,
                 rebalance_classes=False, **kwargs):

        r"""Initialize tf.data.TFRdatasets.

        Parameters
        ----------
        meta : MetaData
            Instance of MetaData, output of mneflow.utils.produce_tfrecords.
            See mneflow.utils.produce_tfrecords and mneflow.MetaData for details.

        train_batch : int, None, optional
            Training mini-batch size. Defaults to 50. If None equals to the
            whole training set size

        test_batch : int, None, optional
            Training mini-batch size. Defaults to None. If None equals to the
            whole test/validation set size

        split : bool
            Whether to split dataset into training and validation folds based
            on h_params['folds']. Defaults to True. Can be False if dataset is
            imported for evaluationg performance on the held-out set or
            vizualisations

        class_subset : list of int
            Pick a susbet of the classes. Example in 5-class clalssification
            problem class_subset=[0, 2, 4] will filter the dataset to
            discriminate between these classes, without changing the parameters
            of the whole dataset (e.g. y_shape=5)

        pick_channels : array of int, optional
            Pick a subset of channels. Defaults to None, in which case all
            channels are used (or, if already set in ``meta.data``, the
            previously stored ``channel_subset`` is kept).

        decim : int, optional
            Apply decimation in time. Note this feature does not check for
            aliasing effects.

        sample_subset : list of int
            NOT IMPLEMENTED

        crop : tuple, optional
            Indices along the time axis to crop. Can be int or None.

        rebalance_classes : bool
            Apply rejection sampling to oversample underrepresented classes.
            Defaults to False.

        **kwargs : dict
            Additional keyword arguments. Accepted for interface
            compatibility with other constructors; currently unused.

        """
        self.h_params = meta.data
        if pick_channels or not 'channel_subset' in self.h_params.keys():
            self.h_params['channel_subset'] = pick_channels
        if np.any(class_subset) or not 'class_subset' in self.h_params.keys():
            self.h_params['class_subset'] = class_subset
        if np.any(sample_subset) or not 'sample_subset' in self.h_params.keys():
            self.h_params['sample_subset'] = sample_subset
        if decim or not 'decim' in self.h_params.keys():
            self.h_params['decim'] = decim
        if crop or not 'crop' in self.h_params.keys():
            self.h_params['crop'] = crop
        if not 'train_batch' in self.h_params.keys() or self.h_params['train_batch'] == None:
            self.h_params['train_batch'] = train_batch
        if not test_batch:
            test_batch = train_batch

        if not 'test_batch' in self.h_params.keys() or self.h_params['test_batch'] == None:
            self.h_params['test_batch'] = test_batch
        if rebalance_classes or not 'rebalance_classes' in self.h_params.keys():
            self.h_params['rebalance_classes'] = rebalance_classes

        self.y_shape = self.h_params['y_shape']
        self.train, self.val = self._build_dataset(self.h_params['train_paths'],
                                                   train_batch=self.h_params['train_batch'],
                                                   test_batch=test_batch,
                                                   split=True, val_fold_ind=0,
                                                   rebalance_classes=self.h_params['rebalance_classes'])
        if len(self.h_params['test_paths']) > 0:
            self.test = self._build_dataset(self.h_params['test_paths'],
                                            train_batch=self.h_params['train_batch'],
                                            test_batch=test_batch,
                                            split=False,
                                            rebalance_classes=self.h_params['rebalance_classes'])
        meta.update(data=self.h_params)



    def _build_dataset(self, path, split=True,
                       train_batch=100, test_batch=None,
                       repeat=True, val_fold_ind=0, holdout=False,
                       rebalance_classes=False):

        """Produce a tf.Dataset object and apply preprocessing
        functions if specified.

        Parameters
        ----------
        path : str or list of str
            Path(s) to the TFRecords file(s) to load.

        split : bool, optional
            Whether to split the resulting dataset into training and
            validation folds using ``self.h_params['folds']``.
            Defaults to True.

        train_batch : int, optional
            Training mini-batch size. Defaults to 100.

        test_batch : int, None, optional
            Test/validation mini-batch size. Defaults to None, in
            which case the validation batch size falls back to
            ``train_batch`` when a ``sample_subset`` is set, or to
            the full size of the validation fold otherwise.

        repeat : bool, optional
            Reserved for future use (dataset repetition control).
            Defaults to True. Currently unused.

        val_fold_ind : int, optional
            Index of the fold (within ``self.h_params['folds']``) to
            use as the validation fold. Defaults to 0.

        holdout : bool, optional
            Reserved for future use (held-out set handling). Defaults
            to False. Currently unused.

        rebalance_classes : bool, optional
            Apply rejection sampling to oversample underrepresented
            classes. Defaults to False.

        Returns
        -------
        train_dataset : tf.data.Dataset
            Preprocessed training dataset. Only returned if
            ``split`` is True.

        val_dataset : tf.data.Dataset
            Preprocessed validation dataset. Only returned if
            ``split`` is True.

        dataset : tf.data.Dataset
            Preprocessed dataset. Only returned if ``split`` is
            False.

        """
        # import and process parent dataset
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(self._parse_function)

        #Define batch sizes
        if split == True:
            train_folds = []
            val_folds = []
            train_inds = []
            val_inds = []

            # split into training and validation folds for each tfrecord file
            # and concatenate

            for i, tfrecord_folds in enumerate(self.h_params['folds']):
                f = tfrecord_folds.copy()
                vf = f.pop(val_fold_ind)
                val_folds.extend(vf)
                train_folds.extend(np.concatenate(f))
                if 'indices' in self.h_params.keys():
                    inds = self.h_params['indices'][i].copy()
                    v_inds = inds.pop(val_fold_ind)

                    val_inds.extend(v_inds)
                    train_inds.extend(np.concatenate(inds))
                    self.val_inds = np.array(val_inds)
                    self.train_inds = np.array(train_inds)
                #print("datafile: {} iter: {} val: {} train: {}".format(i, val_fold_ind, len(val_folds), len(train_folds)))

            self.val_fold = np.array(val_folds)
            self.train_fold = np.array(train_folds)


            self.training_batch = train_batch
            if not test_batch:
                if 'sample_subset' in self.h_params.keys() and self.h_params['sample_subset'] is not None:
                    self.validation_batch = train_batch
                else:
                    self.validation_batch = len(self.val_fold)
            else:
                self.validation_batch = test_batch

            self.validation_steps = max(1, len(self.val_fold) // self.validation_batch)
            self.training_steps = max(1, len(self.train_fold) // self.training_batch)

        else:

            self.test_batch = train_batch

            self.test_steps = max(1, self.test_batch // self.test_batch)

        if 'sample_subset' in self.h_params.keys() and self.h_params['sample_subset'] is not None:

            batch_range = np.arange(train_batch)
            self.sample_subset = np.stack(np.meshgrid(batch_range,
                                                      self.h_params['sample_subset'],
                                                      self.h_params['sample_subset'],
                                                      indexing='ij'),
                                                      axis=-1)

            print('Picking {} sample points {}'.format(len(self.h_params['sample_subset']),
                                                    self.h_params['sample_subset']))


        if 'timepoint_subset' in self.h_params.keys() and self.h_params['timepoint_subset'] is not None:



            self.timepoint_subset = self.h_params['timepoint_subset']
            print('Picking {} timepoints {}'.format(len(self.h_params['timepoint_subset']),
                                                    self.h_params['timepoint_subset']))

        if split:

            train_dataset = dataset.filter(self._cv_train_fold_filter)
            val_dataset =  dataset.filter(self._cv_val_fold_filter)

            train_dataset = self._preprocess(train_dataset, dataset_type='train')
            val_dataset = self._preprocess(val_dataset, dataset_type='val')

            return train_dataset, val_dataset

        else:
            dataset = self._preprocess(dataset, dataset_type='test')

            return dataset


    def _preprocess(self, dataset, dataset_type='test'):
        """Apply class subsetting, class rebalancing, batching and
        feature subsetting to a raw parsed dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Parsed dataset (as produced by :meth:`_parse_function`)
            to preprocess.

        dataset_type : str {'train', 'val', 'test'}, optional
            Which batch size and shuffling/repeat behaviour to use.
            Defaults to 'test'.

        Returns
        -------
        dataset : tf.data.Dataset
            Batched dataset of ``(X, y)`` pairs, with class subset,
            channel subset, sample subset, and timepoint subset
            applied as configured in ``self.h_params``.

        """
        if self.h_params['class_subset'] is not None and self.h_params['target_type'] == 'int':
            dataset = dataset.filter(self._select_classes)
            dataset = dataset.map(self._select_class_subset)

            subset_ratio = np.sum([v for k,v in self.h_params['class_ratio'].items()
                                   if k in self.h_params['class_subset']])
            ratio_multiplier = 1./subset_ratio
            print("Using class_subset with {} classes:".format(len(self.h_params['class_subset'])))

            print("Subset ratio {:.2f}, Multiplier {:.2f}".format(subset_ratio,
                                                                  ratio_multiplier
                                                                  ))
            cp = {k:v*ratio_multiplier for k,v in self.h_params['class_ratio'].items()
                  if k in self.h_params['class_subset']}

            self.h_params['class_ratio'] = cp
            self.y_shape = (len(self.h_params['class_subset']),)

        if self.h_params['rebalance_classes']:
            dataset = self._resample(dataset)

        if dataset_type == 'train':
            dataset = dataset.shuffle(5).batch(self.training_batch).repeat()
        elif dataset_type == 'val':
            dataset = dataset.shuffle(5).batch(self.validation_batch).repeat()
        elif dataset_type == 'test':
            dataset = dataset.shuffle(5).batch(self.test_batch).repeat()

        if self.h_params['channel_subset'] is not None:
            dataset = dataset.map(self._select_channels)

        if self.h_params['sample_subset'] is not None:
            dataset = dataset.map(self._select_samples)

        if 'timepoint_subset' in self.h_params.keys() and self.h_params['timepoint_subset'] is not None:
            dataset = dataset.map(self._select_timepoints)

        dataset = dataset.map(self._unpack)
        return dataset

    def _select_class_subset(self, example_proto):
        """Pick classes defined in self.h_params['class_subset'] from y.

        Parameters
        ----------
        example_proto : dict
            Parsed example with an ``'y'`` entry to subset.

        Returns
        -------
        example_proto : dict
            The input dict with ``example_proto['y']`` replaced by
            the gathered class subset.

        """
        example_proto['y'] = tf.gather(example_proto['y'],
                                       tf.constant(self.h_params['class_subset']),
                                       axis=0)
        return example_proto

    def _select_channels(self, example_proto):
        """Pick a subset of channels specified by self.h_params['channel_subset'].

        Parameters
        ----------
        example_proto : dict
            Parsed example with an ``'X'`` entry to subset along the
            channel axis (axis 3).

        Returns
        -------
        example_proto : dict
            The input dict with ``example_proto['X']`` replaced by
            the gathered channel subset.

        """
        example_proto['X'] = tf.gather(example_proto['X'],
                                        tf.constant(self.h_params['channel_subset']),
                                        axis=3)
        return example_proto

    def _select_samples(self, example_proto):
        """Pick a subset of samples specified by self.sample_subset.

        Parameters
        ----------
        example_proto : dict
            Parsed example with an ``'X'`` entry to subset using
            ``self.sample_subset``.

        Returns
        -------
        example_proto : dict
            The input dict with ``example_proto['X']`` replaced by
            the gathered sample subset.

        """
        example_proto['X'] = tf.gather_nd(example_proto['X'],
                                        indices=tf.constant(self.sample_subset)
                                            )
        return example_proto


    def class_weights(self):
        """Weights take class proportions into account.

        Returns
        -------
        weights : ndarray
            Per-class weights, inversely proportional to
            ``self.h_params['class_ratio']`` and normalized so that
            their mean weighted by the class ratios equals 1.

        """
        weights = np.array(
                [v for k, v in self.h_params['class_ratio'].items()])
        return (1./np.mean(weights))/weights

    def _select_timepoints(self, example_proto):
        """Downsample data.

        Parameters
        ----------
        example_proto : dict
            Parsed example with an ``'X'`` entry to subset along the
            time axis (axis 2) using ``self.timepoint_subset``.

        Returns
        -------
        example_proto : dict
            The input dict with ``example_proto['X']`` replaced by
            the gathered timepoint subset.

        """
        example_proto['X'] = tf.gather(example_proto['X'],
                                       self.timepoint_subset,
                                       axis=2)

    def _crop(self, example_proto):
        """Crop data on the time axis.

        Parameters
        ----------
        example_proto : dict
            Parsed example with an ``'X'`` entry to crop along the
            time axis (axis 2) using ``self.timepoints``.

        Returns
        -------
        example_proto : dict
            The input dict with ``example_proto['X']`` replaced by
            the cropped data.

        """
        example_proto['X'] = tf.gather(example_proto['X'],
                                        self.timepoints,
                                        axis=2)
    def _parse_function(self, example_proto):
        """Restore data shape from serialized records.

        Parameters
        ----------
        example_proto : tf.Tensor
            A serialized ``tf.train.Example`` read from a TFRecords
            file.

        Returns
        -------
        parsed_features : dict
            Dictionary with keys ``'X'``, ``'y'``, and ``'n'``,
            containing the deserialized input data, target, and
            example index, respectively, with shapes and dtypes
            determined by ``self.h_params['input_type']`` and
            ``self.h_params['target_type']``.

        Raises:
        -------
            ValueError: If the `input_type` does not have the supported
            value.
        """
        keys_to_features = {}

        if self.h_params['input_type'] == 'seq':
            y_sh = (self.h_params['n_seq'], *self.h_params['y_shape'])
        else:
            y_sh = self.h_params['y_shape']

        if self.h_params['input_type'] in ['trials', 'seq', 'continuous', 'fconn']:
            x_sh = (self.h_params['n_seq'], self.h_params['n_t'],
                    self.h_params['n_ch'])
        else:
            raise ValueError('Invalid input type.')

        keys_to_features['X'] = tf.io.FixedLenFeature(x_sh, tf.float32)
        keys_to_features['n'] = tf.io.FixedLenFeature((), tf.int64)

        if self.h_params['target_type'] == 'int':
            keys_to_features['y'] = tf.io.FixedLenFeature(y_sh, tf.int64)

        elif self.h_params['target_type'] in ['float', 'signal']:
            keys_to_features['y'] = tf.io.FixedLenFeature(y_sh, tf.float32)

        else:
            raise ValueError('Invalid target type.')

        parsed_features = tf.io.parse_single_example(example_proto,
                                                  keys_to_features)
        return parsed_features

    def _select_classes(self, sample):
        """Filter examples to keep only those in self.h_params['class_subset'].

        Parameters
        ----------
        sample : dict
            Parsed example with a ``'y'`` entry (one-hot encoded
            target) to test against the class subset.

        Returns
        -------
        out : tf.Tensor
            Scalar boolean tensor, True if ``sample['y']`` belongs to
            one of the classes in ``self.h_params['class_subset']``
            (or if no class subset is set), False otherwise.

        """
        if self.h_params['class_subset']:
            # TODO: fix subsetting
            onehot_subset = _onehot(self.h_params['class_subset'],
                                    n_classes=self.h_params['y_shape'][0])
            subset = tf.constant(onehot_subset, dtype=tf.int64)
            out = tf.reduce_any(tf.reduce_all(tf.equal(sample['y'], subset), axis=1))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _cv_train_fold_filter(self, sample):
        """Filter examples to keep only those whose index falls in self.train_fold.

        Parameters
        ----------
        sample : dict
            Parsed example with an ``'n'`` entry (example index) to
            test against ``self.train_fold``.

        Returns
        -------
        out : tf.Tensor
            Scalar boolean tensor, True if ``sample['n']`` is in
            ``self.train_fold`` (or if ``self.train_fold`` is empty),
            False otherwise.

        """
        if np.any(self.train_fold):
            subset = tf.constant(self.train_fold, dtype=tf.int64)
            out = tf.reduce_any(tf.equal(sample['n'], subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _cv_val_fold_filter(self, sample):
        """Filter examples to keep only those whose index falls in self.val_fold.

        Parameters
        ----------
        sample : dict
            Parsed example with an ``'n'`` entry (example index) to
            test against ``self.val_fold``.

        Returns
        -------
        out : tf.Tensor
            Scalar boolean tensor, True if ``sample['n']`` is in
            ``self.val_fold`` (or if ``self.val_fold`` is empty),
            False otherwise.

        """
        if np.any(self.val_fold):
            subset = tf.constant(self.val_fold, dtype=tf.int64)
            out = tf.reduce_any(tf.equal(sample['n'], subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _unpack(self, sample):
        """Extract the input data and target from a parsed example.

        Parameters
        ----------
        sample : dict
            Parsed example with ``'X'`` and ``'y'`` entries.

        Returns
        -------
        X : tf.Tensor
            Input data, ``sample['X']``.

        y : tf.Tensor
            Target, ``sample['y']``.

        """
        return sample['X'], sample['y']#, sample['n']

    def assign_bin(self, sample):
        """Assign each sample to a bin based on its target value.

        Parameters
        ----------
        sample : dict
            Parsed example with a ``'y'`` entry (regression target)
            to bin according to ``self.h_params['bins']``.

        Returns
        -------
        bin_id : tf.Tensor
            Index of the bin (clipped to
            ``[0, self.h_params['n_bins'] - 1]``) that ``sample['y']``
            falls into.

        """
        # Assign each sample to a bin
        bin_id = tf.searchsorted(self.h_params['bins'],
                                 sample['y'], side='left')
        bin_id = tf.clip_by_value(bin_id, 0, self.h_params['n_bins'] - 1)[0]
        return bin_id

    def _resample(self, dataset):
        """Rejection-resample a dataset to balance class/bin proportions.

        For classification targets (``target_type == 'int'``),
        resamples according to ``self.h_params['class_ratio']`` using
        :func:`class_func` to derive the class of each example. For
        regression targets (``target_type == 'float'``), bins the
        target values (see :meth:`assign_bin`) using
        ``self.h_params['bins']`` derived from
        ``self.h_params['class_ratio']`` and resamples across bins.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Dataset of parsed examples to resample.

        Returns
        -------
        balanced_ds : tf.data.Dataset
            Rejection-resampled dataset with (approximately) uniform
            class/bin distribution, unwrapped back to plain examples.

        """
        #print("Oversampling")
        n_classes = len(self.h_params['class_ratio'].items())
        target_dist = 1./n_classes*np.ones(n_classes)
        empirical_dist = [v for k, v in self.h_params['class_ratio'].items()]
        print(self.h_params['class_ratio'], n_classes)

        if self.h_params['target_type'] == 'int':
            #Classification case
            resample_ds = dataset.rejection_resample(class_func,
                                                     target_dist=target_dist,
                                                     initial_dist=empirical_dist)

        elif self.h_params['target_type'] == 'float':

            self.h_params['bins'] = np.array(list(self.h_params['class_ratio'].values())).astype(np.float32)
            target_dist = 1./len(self.h_params['bins'])*np.ones(len(self.h_params['bins']))
            print('n_bins {}, len(bins): {}'.format(self.h_params['n_bins'],
                                                    len(self.h_params['bins'])))

            resample_ds = dataset.rejection_resample(self.assign_bin,
                                                     target_dist=target_dist,
                                                     #initial_dist=empirical_dist
                                                     )

        balanced_ds = resample_ds.map(lambda y, xy: xy)
        new_dist = {k: target_dist[0]
                    for k in self.h_params['class_ratio'].keys()}
        return balanced_ds

def class_func(sample):
    """Extract the class index from a one-hot encoded target.

    Used as the ``class_func`` callback for
    ``tf.data.Dataset.rejection_resample`` in :meth:`Dataset._resample`.

    Parameters
    ----------
    sample : dict
        Parsed example with a ``'y'`` entry (one-hot encoded class
        label).

    Returns
    -------
    class_index : tf.Tensor
        Index of the argmax along the last axis of ``sample['y']``.

    """
    return tf.argmax(sample['y'], -1)
