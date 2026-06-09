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
    """TFRecords dataset from TFRecords files using the metadata."""

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

        sample_subset : list of int
            NOT IMPLEMENTED

        pick_channels : array of int
            Pick a subset of channels

        decim : int
            Apply decimation in time. Note this feature does not check for
            aliasing effects.

        crop : tuple, optional
            Indices along the time axis to crop. Can be int or None.

        rebalance_classes : bool
            Apply rejection sampling to oversample underrepresented classes.
            Defaults to False.

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
        if train_batch or not 'train_batch' in self.h_params.keys():
            self.h_params['train_batch'] = train_batch
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

        """
        # import and process parent dataset
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(self._parse_function)

        #Define batch sizes
        if split == True:
            train_folds = []
            val_folds = []
            #split into training and validation folds

            for i, fold in enumerate(self.h_params['folds']):
                f = fold.copy()
                vf = f.pop(val_fold_ind)
                val_folds.extend(vf)
                train_folds.extend(np.concatenate(f))

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

            print(self.training_batch, self.validation_batch)

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
            dataset = dataset.shuffle(5).batch(self.test_batch)

        if self.h_params['channel_subset'] is not None:
            dataset = dataset.map(self._select_channels)

        if self.h_params['sample_subset'] is not None:
            dataset = dataset.map(self._select_samples)

        if 'timepoint_subset' in self.h_params.keys() and self.h_params['timepoint_subset'] is not None:
            dataset = dataset.map(self._select_timepoints)

        dataset = dataset.map(self._unpack)
        return dataset

    def _select_class_subset(self, example_proto):
        """Pick classes defined in self.h_params['class_subset'] from y"""
        example_proto['y'] = tf.gather(example_proto['y'],
                                       tf.constant(self.h_params['class_subset']),
                                       axis=0)
        return example_proto

    def _select_channels(self, example_proto):
        """Pick a subset of channels specified by self.channel_subset."""
        example_proto['X'] = tf.gather(example_proto['X'],
                                        tf.constant(self.h_params['channel_subset']),
                                        axis=3)
        return example_proto

    def _select_samples(self, example_proto):
        """Pick a subset of channels specified by self.channel_subset."""
        example_proto['X'] = tf.gather_nd(example_proto['X'],
                                        indices=tf.constant(self.sample_subset)
                                            )
        return example_proto


    def class_weights(self):
        """Weights take class proportions into account."""
        weights = np.array(
                [v for k, v in self.h_params['class_ratio'].items()])
        return (1./np.mean(weights))/weights

    def _select_timepoints(self, example_proto):
        """Downsample data."""
        example_proto['X'] = tf.gather(example_proto['X'],
                                       self.timepoint_subset,
                                       axis=2)

    def _crop(self, example_proto):
        """Crop data on the time axis."""
        example_proto['X'] = tf.gather(example_proto['X'],
                                        self.timepoints,
                                        axis=2)
    def _parse_function(self, example_proto):
        """Restore data shape from serialized records.

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
        """Pick a subset of classes specified in self.h_params['class_subset']."""
        if self.h_params['class_subset']:
            # TODO: fix subsetting
            onehot_subset = _onehot(self.h_params['class_subset'],
                                    n_classes=self.h_params['y_shape'][0])
            #print(onehot_subset)
            subset = tf.constant(onehot_subset, dtype=tf.int64)
            out = tf.reduce_any(tf.reduce_all(tf.equal(sample['y'], subset), axis=1))
            # if out == False:
            #     print("X")
            # else:
            #     print("+")

            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _cv_train_fold_filter(self, sample):
        """Pick a subset of classes specified in self.h_params['class_subset']."""
        if np.any(self.train_fold):
            subset = tf.constant(self.train_fold, dtype=tf.int64)
            #print(subset)
            out = tf.reduce_any(tf.equal(sample['n'], subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _cv_val_fold_filter(self, sample):
        """Pick a subset of classes specified in self.h_params['class_subset']."""
        if np.any(self.val_fold):
            subset = tf.constant(self.val_fold, dtype=tf.int64)
            out = tf.reduce_any(tf.equal(sample['n'], subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _unpack(self, sample):
        return sample['X'], sample['y']#, sample['n']

    def assign_bin(self, sample):
        # Assign each sample to a bin

        bin_id = tf.searchsorted(self.h_params['bins'],
                                 sample['y'], side='left')
        bin_id = tf.clip_by_value(bin_id, 0, self.h_params['n_bins'] - 1)[0]
        return bin_id

    def _resample(self, dataset):
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
            #using width-based resampling
            #self.h_params['bins'] = np.array(list(self.h_params['orig_classees'].values())).astype(np.float32)[1:-1]
            #self.h_params['n_bins'] =
            #using percentile-based resampling

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
        #print("New class ratio: ", new_dist)
        #self.h_params['class_ratio'] = new_dist
        return balanced_ds

def class_func(sample):
    return tf.argmax(sample['y'], -1)




# def _onehot(y, n_classes=False):
#     if not n_classes:
#         """Create one-hot encoded labels."""
#         n_classes = len(set(y))
#     out = np.zeros((len(y), n_classes))
#     for i, ii in enumerate(y):
#         out[i][ii] += 1
#     return out.astype(int)