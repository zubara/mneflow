#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines mneflow.Dataset object.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Dataset(object):
    """TFRecords dataset from TFRecords files using the metadata."""

    def __init__(self, h_params, train_batch=200, class_subset=None,
                 combine_classes=False, pick_channels=None, decim=None):

        r"""Initialize tf.data.TFRdatasets.

        Parameters
        ----------
        h_params : dict
            Metadata file, output of mneflow.utils.produce_tfrecords.
            See mneflow.utils.produce_tfrecords for details.

        train_batch : int, optional
            Training mini-batch size. Defaults to 200.

        class_subset : NoneType, list of int, optional
            Pick a subset of classes from the dataset. Note that class
            labels produced by mneflow are always defined as integers
            in range [0 - n_classes). See meta['orig_classes'] for
            mapping between {new_class:old_class}.
            If None, all classes are used.
            Defaults to None.

        combine_classes : list, optional
            Not Implemented, optional

        pick_channels : NoneType, ndarray of int, optional
            Indices of a subset of channels to pick for analysis.
            If None, all classes are used.  Defaults to None.

        decim : NoneType, int, optional
            Decimation factor. Defaults to None.

        """
        self.h_params = h_params
        self.channel_subset = pick_channels
        self.class_subset = class_subset
        self.decim = decim
        self.h_params['train_batch'] = train_batch

        self.train = self._build_dataset(self.h_params['train_paths'],
                                         n_batch=train_batch)
        self.val = self._build_dataset(self.h_params['val_paths'],
                                       n_batch=train_batch)
        if 'test_paths' in self.h_params.keys():
            if len(self.h_params['test_paths']):
                self.test = self._build_dataset(self.h_params['test_paths'],
                                                n_batch=None)
        if isinstance(self.decim, int):
            self.h_params['n_t'] /= self.decim

    def _build_dataset(self, path, n_batch=None):
        """Produce a tf.Dataset object and apply preprocessing
        functions if specified.
        """



        dataset = tf.data.TFRecordDataset(path)

        dataset = dataset.map(self._parse_function)

        if self.channel_subset is not None:
            dataset = dataset.map(self._select_channels)

        if self.class_subset is not None:
            dataset = dataset.filter(self._select_classes)

        if self.decim is not None:
            print('decimating')
            self.timepoints = tf.constant(
                    np.arange(0, self.h_params['n_t'], self.decim))
            dataset = dataset.map(self._decimate)


        if n_batch:
            dataset = dataset.batch(batch_size=n_batch).repeat()
        else:
            dataset = dataset.repeat()

        dataset = dataset.map(self._unpack)
        dataset.n_samples = self._get_n_samples(path)
        print(dataset.n_samples)
        return dataset

    def _select_channels(self, example_proto):
        """Pick a subset of channels specified by self.channel_subset."""
        example_proto['X'] = tf.gather(example_proto['X'],
                                       tf.constant(self.channel_subset),
                                       axis=0)
        return example_proto

    def class_weights(self):
        """Weights take class proportions into account."""
        weights = np.array(
                [v for k, v in self.h_params['class_proportions'].items()])
        return 1./(weights*float(len(weights)))

    def _decimate(self, example_proto):
        """Downsample data."""
        example_proto['X'] = tf.gather(example_proto['X'],
                                       self.timepoints,
                                       axis=-1)
        return example_proto

    def _get_n_samples(self, path):
        """Count number of samples in TFRecord files specified by path."""
        ns = 0
        if isinstance(path, (list, tuple)):
            for fn in path:
                for record in tf.python_io.tf_record_iterator(fn):
                    ns += 1
        elif isinstance(path, str):
            for record in tf.python_io.tf_record_iterator(path):
                ns += 1
        return ns

    def _parse_function(self, example_proto):
        """Restore data shape from serialized records.

        Raises:
        -------
            ValueError: If the `input_type` does not have the supported
            value.
        """
        keys_to_features = {}

        if self.h_params['input_type'] in ['trials', 'iid']:
            x_sh = (self.h_params['n_ch'], self.h_params['n_t'])
            y_sh = self.h_params['y_shape']

        elif self.h_params['input_type'] in ['seq']:
            x_sh = (self.h_params['n_seq'], self.h_params['n_ch'],
                    self.h_params['n_t'])
            y_sh = (self.h_params['y_shape'])
        else:
            raise ValueError('Invalid input type.')

        keys_to_features['X'] = tf.io.FixedLenFeature(x_sh, tf.float32)
        if self.h_params['target_type'] == 'int':
            keys_to_features['y'] = tf.io.FixedLenFeature(y_sh, tf.int64)

        elif self.h_params['target_type'] == 'float':
            keys_to_features['y'] = tf.io.FixedLenFeature(y_sh, tf.float32)

        else:
            raise ValueError('Invalid target type.')

        parsed_features = tf.parse_single_example(example_proto,
                                                  keys_to_features)
        return parsed_features

    def _select_classes(self, sample):
        """Pick a subset of classes specified in self.class_subset."""
        if self.class_subset:
            # TODO: fix subsetting
            onehot_subset = _onehot(self.class_subset,
                                    n_classes=self.h_params['y_shape'])
            #print(onehot_subset)
            subset = tf.constant(onehot_subset, dtype=tf.int64)
            out = tf.reduce_any(tf.equal(tf.argmax(sample['y']), subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _unpack(self, sample):
        return sample['X'], sample['y']


def _onehot(y, n_classes=False):
    if not n_classes:
        """Create one-hot encoded labels."""
        n_classes = len(set(y))
    out = np.zeros((len(y), n_classes))
    for i, ii in enumerate(y):
        out[i][ii] += 1
    return out.astype(int)