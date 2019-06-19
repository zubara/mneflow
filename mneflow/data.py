#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:58:39 2019

@author: zubarei1
"""
import tensorflow as tf
import numpy as np


class Dataset(object):
    """TFRecords dataset from TFRecords files using the metadata.

        """

    def __init__(self, h_params, train_batch=200, class_subset=None,
                 combine_classes=False, pick_channels=None, decim=None):

        """Initialize tf.data.TFRdatasets

        Parameters:
        ----------
        h_params : dict
                    metadata file, output of mneflow.utils.produce_tfrecords.
                    See mneflow.utils.produce_tfrecords for details.

        train_batch : int, optional
                    Training mini-batch size. Deafults to 200

        class_subset : NoneType, list of int, optional
                    Pick a subset of classes from the dataset. Note that class
                    labels produced by mneflow are always defined as integers
                    in range [0 - n_classes). See meta['orig_classes'] for
                    mapping between {new_class:old_class}.
                    If None, all classes are used.  Defaults to None.

        combine_classes : Not Implemented, optional

        pick_channels : NoneType, ndarray of int, optional
                    Indices of a subset of channels to pick for analysis.
                    If None, all classes are used.  Defaults to None.

        decim : NoneType, int, optional
                Decimation factor. Defaults to None
        """
        self.h_params = h_params
        self.channel_subset = pick_channels
        self.class_subset = class_subset
        self.decim = decim
        self.train = self.build_dataset(self.h_params['train_paths'],
                                        n_batch=train_batch)
        self.val = self.build_dataset(self.h_params['val_paths'],
                                      n_batch=None)
        if isinstance(self.decim, int):
            self.h_params['n_t'] /= self.decim

    def build_dataset(self, path, n_batch=None):
        """Produce a tf.Dataset object and apply preprocessing functions
        if specified"""
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(self._parse_function)
        if not self.channel_subset is None:
            dataset = dataset.map(self.select_channels)
        if not self.class_subset is None:
            dataset = dataset.filter(self.select_classes)
        if not self.decim is None:
            print('decimating')
            self.timepoints = tf.constant(np.arange(0, self.h_params['n_t'], self.decim))
            dataset = dataset.map(self.decimate)
        if n_batch:
            dataset = dataset.batch(n_batch).repeat()
        else:
            ds_size = self.get_n_samples(path)
            dataset = dataset.batch(ds_size).repeat()
        dataset = dataset.map(self.unpack)
        return dataset

    def select_channels(self, example_proto):
        """Pick a subset of channels specified by self.channel_subset"""
        example_proto['X'] = tf.gather(example_proto['X'],
                                       tf.constant(self.channel_subset),
                                       axis=0)
        return example_proto

#    def global_scale(self,example_proto):
#        example_proto['X'] -= tf.reduce_mean(example_proto['X'],axis = -1,
#        return example_proto

    def decimate(self, example_proto):
        """Downsample data"""
        print(example_proto['X'].shape)
        example_proto['X'] = tf.gather(example_proto['X'],
                                       self.timepoints,
                                       axis=-1)
        print(example_proto['X'].shape)
        return example_proto

    def get_n_samples(self, path):
        """Count number of samples in TFRecord files specified by path"""
        ns = 0
        for fn in path:
            for record in tf.python_io.tf_record_iterator(fn):
                ns += 1
        return ns

    def _parse_function(self, example_proto):
        """Restore data shape from serialized records"""
        if self.h_params['task'] == 'classification':
            keys_to_features = {'X': tf.FixedLenFeature((self.h_params['n_ch'],
                                                        self.h_params['n_t']),
                                                        tf.float32),
                                'y': tf.FixedLenFeature((),
                                                        tf.int64,
                                                        default_value=0)}
        else:
            keys_to_features = {'X': tf.FixedLenFeature((self.h_params['n_ch'],
                                                         self.h_params['n_t']),
                                                        tf.float32),
                                'y': tf.FixedLenFeature(self.h_params['y_shape'],
                                                        tf.float32)}
        parsed_features = tf.parse_single_example(example_proto,
                                                  keys_to_features)
        return parsed_features

    def select_classes(self, sample):
        """Picks a subset of classes specified in self.class_subset"""
        if self.class_subset:
            subset = tf.constant(self.class_subset, dtype=tf.int64)
            out = tf.reduce_any(tf.equal(sample['y'], subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def unpack(self, sample):
        return sample['X'], sample['y']
