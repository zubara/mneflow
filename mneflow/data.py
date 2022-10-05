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

    def __init__(self, h_params, train_batch=200, test_batch=None, split=True,
                 class_subset=None,
                 #combine_classes=False, 
                 pick_channels=None, decim=None,
                 rebalance_classes=False
                 ):

        r"""Initialize tf.data.TFRdatasets.

        Parameters
        ----------
        h_params : dict
            Metadata file, output of mneflow.utils.produce_tfrecords.
            See mneflow.utils.produce_tfrecords for details.

        train_batch : int, None, optional
            Training mini-batch size. Defaults to 200. If None equals to the
            whole training set size

        test_batch : int, None, optional
            Training mini-batch size. Defaults to None. If None equals to the
            whole test/validation set size

        split : bool
            Whether to split dataset into training and validation folds based
            on h_params['folds']. Defaults to True. Can be False if dataset is
            imported for evaluationg performance on the held-out set or
            vizualisations


        """
        self.h_params = h_params
        self.channel_subset = pick_channels
        self.class_subset = class_subset
        self.decim = decim
        self.h_params['train_batch'] = train_batch


        #%%%%%%%%%%%%

        self.train, self.val = self._build_dataset(self.h_params['train_paths'],
                                                   train_batch=train_batch,
                                                   test_batch=test_batch,
                                                   split=True, val_fold_ind=0,
                                                   rebalance_classes=rebalance_classes)
        if len(self.h_params['test_paths']) > 0:
            self.test = self._build_dataset(self.h_params['test_paths'],
                                            train_batch=train_batch,
                                            test_batch=test_batch,
                                            split=False,
                                            rebalance_classes=rebalance_classes)


        # #self.val = self._build_dataset(self.h_params['val_paths'],
        # #                               n_batch=test_batch)
        # if 'test_paths' in self.h_params.keys():
        #     if len(self.h_params['test_paths']):
        #         self.test = self._build_dataset(self.h_params['test_paths'],
        #                                         n_batch=test_batch)
        #%%%%%%%%%%%%%

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

        if self.channel_subset is not None:
            dataset = dataset.map(self._select_channels)

        if self.class_subset is not None and self.h_params['target_type'] == 'int':
            dataset = dataset.filter(self._select_classes)
            subset_ratio = np.sum([v for k,v in self.h_params['class_ratio'].items()
                                   if k in self.class_subset])
            ratio_multiplier = 1./subset_ratio
            print("Subset ratio {:.2f}, Multiplier {:.2f}".format(subset_ratio,
                                                                  ratio_multiplier
                                                                  ))
            cp = {k:v*ratio_multiplier for k,v in self.h_params['class_ratio'].items() 
                  if k in self.class_subset}
            
            self.h_params['class_ratio'] = cp
            #print(self.h_params['class_ratio'])
            
        if self.decim is not None:
            print('decimating')

            self.timepoints = tf.constant(
                    np.arange(0, self.h_params['n_t'], self.decim))

            self.h_params['n_t'] = len(self.timepoints)
            dataset = dataset.map(self._decimate)

        #TODO: test set case

        if split:
            train_folds = []
            val_folds = []
            #split into training and validation folds

            for i, fold in enumerate(self.h_params['folds']):
                f = fold.copy()
                vf = f.pop(val_fold_ind)
                val_folds.extend(vf)
                train_folds.extend(np.concatenate(f))
                #print("datafile: {} iter: {} val: {} train: {}".format(i, val_fold_ind, len(val_folds), len(train_folds)))


            self.val_fold = np.array(val_folds)
            self.train_fold = np.array(train_folds)

            # ovl = 0
            # for si in self.train_fold:
            #     if si in self.val_fold:
            #         ovl += 1
            # print('OVERLAP: ', ovl)
            #print(len(np.concatenate(folds)))
            #print("Train fold:", self.train_fold, self.train_fold.shape)
            #print("val fold:", self.val_fold, self.val_fold.shape)
            #self.train_fold = np.concatenate(self.train_fold)

            train_dataset = dataset.filter(self._cv_train_fold_filter)
            val_dataset =  dataset.filter(self._cv_val_fold_filter)
            
            if rebalance_classes:
                train_dataset = self._resample(train_dataset)
                val_dataset = self._resample(val_dataset)

            #batch
            if not test_batch:
                test_batch = len(self.val_fold)

            self.validation_steps = max(1, len(self.val_fold)//test_batch)
            self.training_steps = max(1, len(self.train_fold)//train_batch)
            self.validation_batch = test_batch
            self.training_batch = train_batch

            val_dataset = val_dataset.shuffle(5).batch(test_batch).repeat()
            val_dataset.batch_size = test_batch
            train_dataset = train_dataset.shuffle(5).batch(train_batch).repeat()

            train_dataset = train_dataset.map(self._unpack)
            val_dataset = val_dataset.map(self._unpack)

            return train_dataset, val_dataset

        else:
            #print(dataset)
            #batch
            if np.any(['train' in tp for tp in path]):
                size = self.h_params['train_size']
            else:
                size = self.h_params['val_size']
            if not test_batch:
                test_batch = size
                
            if rebalance_classes:
                dataset = self._resample(dataset)
                
            dataset = dataset.shuffle(5).batch(test_batch)#.repeat()
            dataset.batch = test_batch


            self.test_batch = test_batch
            self.test_steps = max(1, size // test_batch)
            dataset = dataset.map(self._unpack)
            return dataset#, None
            #else:
            # unsplit datasets are used for visuzalization and evaluation
            # if batching is not specified the whole set is used as batch

        #     val_size = self.dataset.h_params['val_size']
        #     self.validation_steps =  val_size // val_batch)
        # else:
        #     self.validation_steps = 1


            # print(dataset)

            # else:
            #     test_batch = self.h_params['val_size']
            #     dataset = dataset.shuffle(5).batch(test_batch).repeat()






    def _select_channels(self, example_proto):
        """Pick a subset of channels specified by self.channel_subset."""
        example_proto['X'] = tf.gather(example_proto['X'],
                                        tf.constant(self.channel_subset),
                                        axis=3)
        return example_proto

    def _select_times(self, example_proto):
        """Pick a subset of channels specified by self.channel_subset."""
        example_proto['X'] = tf.gather(example_proto['X'],
                                        tf.constant(self.times),
                                        axis=2)
        return example_proto

    def class_weights(self):
        """Weights take class proportions into account."""
        weights = np.array(
                [v for k, v in self.h_params['class_proportions'].items()])
        return (1./np.mean(weights))/weights

    def _decimate(self, example_proto):
        """Downsample data."""
        example_proto['X'] = tf.gather(example_proto['X'],
                                        self.timepoints,
                                        axis=2)
    #     return example_proto

#    def _get_n_samples(self, path):
#        """Count number of samples in TFRecord files specified by path."""
#        ns = path
#        return ns

    def _parse_function(self, example_proto):
        """Restore data shape from serialized records.

        Raises:
        -------
            ValueError: If the `input_type` does not have the supported
            value.
        """
        keys_to_features = {}

        if self.h_params['input_type'] in ['trials', 'seq', 'continuous']:
            x_sh = (self.h_params['n_seq'], self.h_params['n_t'],
                    self.h_params['n_ch'])
            if self.h_params['input_type'] == 'seq':
                y_sh = (self.h_params['n_seq'], *self.h_params['y_shape'])
            else:
                y_sh = self.h_params['y_shape']

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
        """Pick a subset of classes specified in self.class_subset."""
        if self.class_subset:
            # TODO: fix subsetting
            onehot_subset = _onehot(self.class_subset,
                                    n_classes=self.h_params['y_shape'][0])
            #print(onehot_subset)
            subset = tf.constant(onehot_subset, dtype=tf.int64)
            out = tf.reduce_any(tf.equal(tf.argmax(sample['y']), subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _cv_train_fold_filter(self, sample):
        """Pick a subset of classes specified in self.class_subset."""
        if np.any(self.train_fold):
            subset = tf.constant(self.train_fold, dtype=tf.int64)
            #print(subset)
            out = tf.reduce_any(tf.equal(sample['n'], subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _cv_val_fold_filter(self, sample):
        """Pick a subset of classes specified in self.class_subset."""
        if np.any(self.val_fold):
            subset = tf.constant(self.val_fold, dtype=tf.int64)
            out = tf.reduce_any(tf.equal(sample['n'], subset))
            return out
        else:
            return tf.constant(True, dtype=tf.bool)

    def _unpack(self, sample):
        return sample['X'], sample['y']#, sample['n']
    
           
    
    def _resample(self, dataset):
        #print("Oversampling")
        
        n_classes = len(self.h_params['class_ratio'].items())
        print(n_classes)
        target_dist = 1./n_classes*np.ones(n_classes)
        empirical_dist = [v for k, v in self.h_params['class_ratio'].items()]
        resample_ds = dataset.rejection_resample(class_func, 
                                                 target_dist=target_dist, 
                                                 initial_dist=empirical_dist)
        balanced_ds = resample_ds.map(lambda y, xy: xy)
        new_dist = {k: target_dist[0] 
                    for k in self.h_params['class_ratio'].keys()}
        print("New class ratio: ", new_dist)
        self.h_params['class_ratio'] = new_dist
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