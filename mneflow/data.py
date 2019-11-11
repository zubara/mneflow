#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines mneflow.Dataset object
"""
import tensorflow as tf
import numpy as np


class Dataset(object):
    """TFRecords dataset from TFRecords files using the metadata.

        """

    def __init__(self, h_params, train_batch=200, class_subset=None,
                 combine_classes=False, pick_channels=None, decim=None):

        """
        Initialize tf.data.TFRdatasets

        Parameters
        ----------
        h_params : dict
                   metadata file, output of mneflow.utils.produce_tfrecords.
                   See mneflow.utils.produce_tfrecords for details.

        train_batch : int, optional
                      Training mini-batch size. Deafults to 200

        class_subset : NoneType, list of int, optional
                       Pick a subset of classes from the dataset. Note that
                       class labels produced by mneflow are always defined as
                       integers in range [0 - n_classes). See
                       meta['orig_classes'] for mapping between
                       {new_class:old_class}. If None, all classes are used.
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

        self.train = self._build_dataset(self.h_params['train_paths'],
                                         n_batch=train_batch)
        self.val = self._build_dataset(self.h_params['val_paths'],
                                       n_batch=None)
        self.test = self._build_dataset(self.h_params['test_paths'],
                                        n_batch=None)
        if isinstance(self.decim, int):
            self.h_params['n_t'] /= self.decim

    def _build_dataset(self, path, n_batch=None):
        """
        Produce a tf.Dataset object and apply preprocessing functions
        if specified.
        """
        if not n_batch:
            if self.h_params['input_type'] in ['seq']:
                n_batch = 1  # each batch contains a single sequence
            else:
                n_batch = self._get_n_samples(path)

        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(self._parse_function)
        if self.channel_subset is not None:
            dataset = dataset.map(self._select_channels)
        if self.class_subset is not None:
            dataset = dataset.filter(self._select_classes)
        if self.decim is not None:
            print('decimating')
            self.timepoints = tf.constant(np.arange(0, self.h_params['n_t'], self.decim))
            dataset = dataset.map(self._decimate)

        dataset = dataset.batch(n_batch).repeat()
        dataset = dataset.map(self._unpack)
        return dataset

    def _select_channels(self, example_proto):
        """Pick a subset of channels specified by self.channel_subset"""
        example_proto['X'] = tf.gather(example_proto['X'],
                                       tf.constant(self.channel_subset),
                                       axis=0)
        return example_proto

#    def global_scale(self,example_proto):
#        example_proto['X'] -= tf.reduce_mean(example_proto['X'],axis = -1,
#        return example_proto
    def class_weights(self):
        weights = np.array([v for k,v in self.h_params['class_proportions'].items()])
        return  1./(weights*float(len(weights)))

    def _decimate(self, example_proto):
        """Downsample data"""
        print(example_proto['X'].shape)
        example_proto['X'] = tf.gather(example_proto['X'],
                                       self.timepoints,
                                       axis=-1)
        print(example_proto['X'].shape)
        return example_proto

    def _get_n_samples(self, path):
        """Count number of samples in TFRecord files specified by path"""
        ns = 0
        if isinstance(path, (list, tuple)):
            for fn in path:
                for record in tf.python_io.tf_record_iterator(fn):
                    ns += 1
        elif isinstance(path, str):
            for record in tf.python_io.tf_record_iterator(path):
                ns += 1
        return ns

#    def parse_seq_record(record):
#        c, f = tf.parse_single_sequence_example(record)
#      return c, f

    def _parse_function(self, example_proto):
        """Restore data shape from serialized records"""
        keys_to_features = {}
        if self.h_params['input_type'] in ['trials', 'iid']:
            keys_to_features['X'] = tf.io.FixedLenFeature((self.h_params['n_ch'],
                                                           self.h_params['n_t']),
                                                          tf.float32)
            if self.h_params['target_type'] == 'int':
                # TODO onehot encoding to utils for classification
                keys_to_features['y'] =  tf.io.FixedLenFeature(self.h_params['y_shape'],
                                                               tf.int64)
            elif self.h_params['target_type'] == 'float':
                keys_to_features['y']  =  tf.io.FixedLenFeature(self.h_params['y_shape'],
                                                                tf.float32)

            parsed_features = tf.parse_single_example(example_proto, keys_to_features)
            print(parsed_features)

        elif self.h_params['input_type'] in ['seq']:

            context_features = {'length': tf.io.FixedLenFeature((), tf.int64, default_value=0)}
            keys_to_features['X'] = tf.io.FixedLenSequenceFeature((self.h_params['n_ch'],
                                                                   self.h_params['n_t'], 1), tf.float32)
            if self.h_params['target_type'] == 'int':
                # TODO onehot encoding to utils for classification
                keys_to_features['y'] =  tf.io.FixedLenSequenceFeature(self.h_params['y_shape'], tf.int64)
            elif self.h_params['target_type'] == 'float':
                keys_to_features['y'] =  tf.io.FixedLenSequenceFeature((self.h_params['y_shape'], 1), tf.float32)

            c, parsed_features = tf.parse_single_sequence_example(example_proto,
                                                                  context_features=context_features,
                                                                  sequence_features=keys_to_features)
            print('------------------------------------')
            # Return back to shape [k, n_seq, n_ch, time]
            parsed_features['X'] = tf.transpose(parsed_features['X'], [3, 0, 1, 2], name='X')
            parsed_features['y'] = tf.transpose(parsed_features['y'], [2, 0, 1], name='y')
            print(parsed_features['X'])
            print(parsed_features['y'])

        return parsed_features

    def _select_classes(self, sample):
        """Picks a subset of classes specified in self.class_subset"""
        if self.class_subset:
            subset = tf.constant(self.class_subset, dtype=tf.int64)
            out = tf.reduce_any(tf.equal(sample['y'], subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _unpack(self, sample):
        return sample['X'], sample['y']
