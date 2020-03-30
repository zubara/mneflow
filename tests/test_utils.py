#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for utils.py.

@author: Gavriela Vranou
"""

import os
import pathlib
import glob
import time
import unittest
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

import mne

from mneflow import utils

# Change current directory to the one this file resides in
os.chdir(pathlib.Path(__file__).parent.absolute())


# --- Auxiliary functions ---
def check_meta_trial_class():
    return dict(train_paths=[], val_paths=[], test_paths=[],
                data_id='', val_size=0, savepath='',
                target_type='', input_type='', fs=0,
                class_proportions=dict(), orig_classes=dict(),
                y_shape=tuple(), n_ch=0, n_t=0, test_size=0)


def check_meta_seq_class():
    return dict(train_paths=[], val_paths=[], test_paths=[],
                data_id='', val_size=0, savepath='', n_seq=0,
                target_type='', input_type='', fs=0,
                class_proportions=dict(), orig_classes=dict(),
                y_shape=tuple(), n_ch=0, n_t=0, test_size=0)


def check_meta_trial_reg():
    return dict(train_paths=[], val_paths=[], test_paths=[],
                data_id='', val_size=0, savepath='',
                target_type='', input_type='', fs=0,
                y_shape=tuple(), n_ch=0, n_t=0, test_size=0)


def check_meta_seq_reg():
    return dict(train_paths=[], val_paths=[], test_paths=[],
                data_id='', val_size=0, savepath='', n_seq=0,
                target_type='', input_type='', fs=0,
                y_shape=tuple(), n_ch=0, n_t=0, test_size=0)


def check_tfrecords(path='./', pattern=''):
    """Iterate over TFRecord files maching the pattern and checks for
    corrupted records.

    Adapted from keras_utils.py"""
    import glob

    total_files = 0
    error_files = 0
    train_files = sorted(glob.glob('%s*%s*.tfrecord' % (path, pattern)))
    for f_i, file in enumerate(train_files):
        try:
            total_files += sum(
                    [1 for _ in tf.python_io.tf_record_iterator(file)])
        except IOError:
            total_files += 1
            error_files += 1

    return total_files, error_files


# --- UNIT TESTS ---
class TestUtils(unittest.TestCase):
    """Unit test class for most functions contained in utils.py file."""

    def test_true(self):
        """Sanity check test #1."""
        self.assertEqual(True, True)

    def test_pass(self):
        """Sanity check test #2."""
        pass

    def test_partition(self):
        """Placeholder for utils.partition"""
        # data = np.arange(10000).reshape(100, 5, 20)
        # data = data.astype(np.float32)
        # events = np.arange(100).reshape(100, 1)

        # idx = [(1, 2), (10, 11)]
        # x1, x2 = utils.partition(data, idx)
        # y1, y2 = utils.partition(events, idx)

    def test_cont_split_indices(self):
        """Unit test for utils.cont_split_indices"""
        data = np.arange(1000).reshape(1, 10, 5, 20)

        idx = utils.cont_split_indices(data, test_size=0.1, test_segments=2)
        self.assertTrue(len(idx) == 2)
        self.assertTrue(np.all([jj < data.shape[-1]
                                for ii in idx for jj in ii]))

    def test_create_example_fif(self):
        from mne.datasets import multimodal

        fname = 'example-epo.fif'
        if not os.path.exists(fname):
            mne.set_log_level(verbose='CRITICAL')
            rname = os.path.join(multimodal.data_path(), 'multimodal_raw.fif')
            raw = mne.io.read_raw_fif(rname)
            cond = raw.acqparser.get_condition(
                raw, condition=['Auditory left', 'Auditory right'])
            epochs_list = [mne.Epochs(raw, **c) for c in cond]
            epochs = mne.concatenate_epochs(epochs_list)
            epochs.save(fname, overwrite=False)
            del raw, epochs, cond, epochs_list

        epochs = mne.epochs.read_epochs(fname, preload=False)
        self.assertTrue(epochs)

    def test_create_example_npz(self):
        fname = 'example_meg.npz'
        if not os.path.exists('example_meg.npz'):
            epochs = mne.read_epochs('example-epo.fif', preload=True)
            data = epochs.get_data()
            events = epochs.events[:, 2]
            np.savez_compressed('example_meg', data=data, events=events)
            del epochs, data, events

        datafile = np.load(fname)
        self.assertTrue(np.any(datafile['data']))
        self.assertTrue(np.any(datafile['events']))

    def test_create_example_mat(self):
        import scipy.io as sio
        fname = 'example_meg.mat'
        if not os.path.exists(fname):
            tmp = np.load('example_meg.npz')
            adict = {}
            adict['data'] = tmp['data']
            adict['events'] = tmp['events']
            sio.savemat(fname, adict)
            del tmp, adict

        datafile = sio.loadmat(fname)
        self.assertTrue(np.any(datafile['data']))
        self.assertTrue(np.any(datafile['events']))

    def test_onehot(self):
        """Unit test for utils._onehot function."""
        y = np.arange(0, 10, dtype='int')
        y_ = utils._onehot(y)
        y_true = np.eye(10, dtype='int')
        np.testing.assert_equal(y_, y_true)

    def test_load_meta_trials(self):
        """Unit test for utils._load_meta function."""
        with self.assertRaises(FileNotFoundError):
            s = utils._load_meta('', '')

        s = utils._load_meta('./', 'example_trials')
        self.assertTrue(isinstance(s, dict))

        tmp = check_meta_trial_class()

        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))

    def test_load_meta_seq(self):
        """Unit test for utils._load_meta function."""

        with self.assertRaises(FileNotFoundError):
            s = utils._load_meta('', '')

        s = utils._load_meta('./', 'example_seq')
        self.assertTrue(isinstance(s, dict))

        tmp = check_meta_seq_class()

        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))

    def test_scale_to_baseline_cont(self):
        """Test on a continuous signal (sigmoid) with std = 1 and mean = 0."""
        from scipy.special import expit
        t = np.linspace(-5, 5, 1000)
        f = 2.56976368*expit(t) - 1.28488184
        X = np.stack([f for _ in range(10)])
        X_ = np.asarray([(ii+1)*X[ii] + (ii+1) for ii in range(10)])
        s = utils.scale_to_baseline(X_.copy())
        # almost_equal since the values are floats
        np.testing.assert_almost_equal(X, s)

    def test_scale_to_baseline_range_crop(self):
        """Baseline is calculated from range and cropped."""
        X = np.stack([[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for _ in range(10)])
        X_ = np.asarray([(ii+1)*X[ii] + 0.5 for ii in range(0, 10)])
        inv = (0, 2)
        s = utils.scale_to_baseline(X_, baseline=inv, crop_baseline=True)
        np.testing.assert_equal(X[..., (inv[1]-1):].shape, s.shape)
        np.testing.assert_equal(X[..., (inv[1]-1):], s)

    def test_scale_to_baseline_range(self):
        """Baseline is calculated from range."""
        X = np.stack([[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for _ in range(10)])
        X_ = np.asarray([(ii+1)*X[ii] + 0.5 for ii in range(0, 10)])
        inv = (0, 2)
        # test without cropping the baseline
        s = utils.scale_to_baseline(X_, baseline=inv, crop_baseline=False)
        np.testing.assert_equal(X.shape, s.shape)
        np.testing.assert_equal(X, s)

    def test_produce_labels(self):
        """Test labels produced from event trigger codes."""
        f = np.arange(1, 10, step=2)
        y = np.stack([f for _ in range(10)]).flatten()
        y = np.random.permutation(y)
        inv, tot, prop, orig = utils.produce_labels(y, return_stats=True)

        self.assertEqual(len(y), tot)
        np.testing.assert_equal(list(prop.keys()), np.arange(5))
        np.testing.assert_equal(list(prop.values()), 0.2*np.ones(5))
        np.testing.assert_equal(list(orig.items()),
                                [(ii, v) for ii, v in enumerate(f)])
        np.testing.assert_equal(inv, np.unique(y, return_inverse=True)[1])

    def test_combine_labels(self):
        """Test label combination."""
        events = np.arange(10) // 2  # range [0, 4]

        # testing single label, invalid label and list of labels
        combine_events = {24: 3, 0: 5, 11: [1, 2]}
        avail_labels = [1, 2, 3]
        new_avail_labels = [11, 24]

        tevents, keep_ind = utils._combine_labels(events, combine_events)
        for a, b in combine_events.items():
            print(a, b)
            idx = [ii for ii, v in enumerate(events) if v in b]
            self.assertTrue(np.all(tevents[idx] == a))
        self.assertTrue(np.all(np.isin(events[keep_ind], avail_labels)))
        self.assertTrue(np.all(events[keep_ind] != 5))
        self.assertTrue(np.all(np.isin(tevents[keep_ind], new_avail_labels)))
        self.assertTrue(np.all(tevents[keep_ind] != 0))


class TestPreprocess(unittest.TestCase):
    """Unit test class for utils.preprocess function."""

    def test_preprocess(self):
        data = np.arange(1000).reshape(10, 5, 20)
        data = data.astype(np.float32)
        events = np.arange(10).reshape(10, 1)

        # No options check
        x1, y1, x2, y2 = utils.preprocess(data, events)
        self.assertEqual(x1.shape[0] + x2.shape[0], data.shape[0])
        self.assertEqual(x1.shape[1:], data.shape[1:])
        self.assertEqual(x2.shape[1:], data.shape[1:])
        self.assertEqual(y1.shape[0] + y2.shape[0], events.shape[0])
        self.assertEqual(y1.shape[1:], events.shape[1:])
        self.assertEqual(y2.shape[1:], events.shape[1:])

    def test_preprocess_trials(self):
        data = np.arange(1000).reshape(10, 5, 20)
        data = data.astype(np.float32)
        events = np.arange(10).reshape(10, 1)

        # trials
        x1, y1, x2, y2 = utils.preprocess(data, events, input_type='trials')
        self.assertEqual(x1.shape[0] + x2.shape[0], data.shape[0])
        self.assertEqual(x1.shape[1:], data.shape[1:])
        self.assertEqual(x2.shape[1:], data.shape[1:])
        self.assertEqual(y1.shape[0] + y2.shape[0], events.shape[0])
        self.assertEqual(y1.shape[1:], events.shape[1:])
        self.assertEqual(y2.shape[1:], events.shape[1:])

    def test_preprocess_seq(self):
        data = np.arange(1000).reshape(10, 5, 20)
        data = data.astype(np.float32)
        events = np.arange(10).reshape(10, 1)

        # seq
        x1, y1, x2, y2 = utils.preprocess(data, events, input_type='seq')
        self.assertEqual(x1.shape[0] + x2.shape[0], data.shape[0])
        self.assertEqual(x1.shape[1:], data.shape[1:])
        self.assertEqual(x2.shape[1:], data.shape[1:])
        self.assertEqual(y1.shape[0] + y2.shape[0], events.shape[0])
        self.assertEqual(y1.shape[1:], events.shape[1:])
        self.assertEqual(y2.shape[1:], events.shape[1:])

    def test_preprocess_scale(self):
        data = np.arange(1000).reshape(10, 5, 20)
        data = data.astype(np.float32)
        events = np.arange(10).reshape(10, 1)

        # scale
        x1, y1, x2, y2 = utils.preprocess(data, events, scale=True)
        self.assertEqual(x1.shape[0] + x2.shape[0], data.shape[0])
        self.assertEqual(x1.shape[1:], data.shape[1:])
        self.assertEqual(x2.shape[1:], data.shape[1:])

        self.assertEqual(y1.shape[0] + y2.shape[0], events.shape[0])
        self.assertEqual(y1.shape[1:], events.shape[1:])
        self.assertEqual(y2.shape[1:], events.shape[1:])

    def test_preprocess_decimate(self):
        data = np.arange(1000).reshape(10, 5, 20)
        data = data.astype(np.float32)
        events = np.arange(10).reshape(10, 1)

        # decimate
        x1, y1, x2, y2 = utils.preprocess(data, events, decimate=2)
        self.assertEqual(x1.shape[0] + x2.shape[0], data.shape[0])
        self.assertEqual(x1.shape[1], data.shape[1])
        self.assertEqual(x1.shape[2], data.shape[2]/2)

        self.assertEqual(x2.shape[1], data.shape[1])
        self.assertEqual(x2.shape[2], data.shape[2]/2)

        self.assertEqual(y1.shape[0] + y2.shape[0], events.shape[0])
        self.assertEqual(y1.shape[1:], events.shape[1:])
        self.assertEqual(y2.shape[1:], events.shape[1:])

    def test_preprocess_segment(self):
        data = np.arange(1000).reshape(10, 5, 20)
        data = data.astype(np.float32)
        events = np.arange(10).reshape(10, 1)

        # segment
        x1, y1, x2, y2 = utils.preprocess(data, events, segment=10)
        self.assertEqual(x1.shape[0] + x2.shape[0], 2*data.shape[0])
        self.assertEqual(x1.shape[1], data.shape[1])
        self.assertEqual(x1.shape[2], data.shape[2]/2)

        self.assertEqual(x2.shape[1], data.shape[1])
        self.assertEqual(x2.shape[2], data.shape[2]/2)

        self.assertEqual(y1.shape[0] + y2.shape[0], 2*events.shape[0])
        self.assertEqual(y1.shape[1:], events.shape[1:])
        self.assertEqual(y2.shape[1:], events.shape[1:])


class TestUtilstf(tf.test.TestCase):
    """Unit tests assessing Tensorflow functionality."""

    def test_true(self):
        """Sanity check test #1."""
        self.assertEqual(True, True)

    def test_pass(self):
        """Sanity check test #2."""
        pass


class TestSplitSets(unittest.TestCase):
    """Unit test class for utils._split_sets function."""

    def test_split_sets_2d_target1d_v1(self):
        """Split numpy arrays into sets."""
        X = np.arange(200).reshape(10, 20)
        y = np.arange(10)  # Note! y is 1D with shape (10, )

        x1, y1, x2, y2 = utils._split_sets(X, y, val=0.5)
        self.assertTrue(np.all(np.isin(x1, X)))
        self.assertTrue(np.all(np.isin(x2, X)))
        self.assertTrue(np.all(np.isin(np.concatenate((x1, x2), axis=0), X)))
        self.assertTrue(np.all(np.isin(X, np.concatenate((x1, x2), axis=0))))
        self.assertEqual(x1.shape[0] + x2.shape[0], X.shape[0])
        self.assertEqual(x1.shape[1], X.shape[1])
        self.assertEqual(x2.shape[1], X.shape[1])

        self.assertTrue(np.all(np.isin(y1, y)))
        self.assertTrue(np.all(np.isin(y2, y)))
        self.assertTrue(np.all(np.isin(np.concatenate((y1, y2), axis=0), y)))
        self.assertTrue(np.all(np.isin(y, np.concatenate((y1, y2), axis=0))))
        self.assertEqual(y1.shape[0] + y2.shape[0], y.shape[0])
        self.assertEqual(y1.shape[1], 1)
        self.assertEqual(y2.shape[1], 1)

    def test_split_sets_2d_target1d_v2(self):
        """Split numpy arrays into sets."""
        X = np.arange(200).reshape(10, 20)
        y = np.arange(10).reshape(10, 1)

        x1, y1, x2, y2 = utils._split_sets(X, y, val=0.5)
        self.assertTrue(np.all(np.isin(x1, X)))
        self.assertTrue(np.all(np.isin(x2, X)))
        self.assertTrue(np.all(np.isin(np.concatenate((x1, x2), axis=0), X)))
        self.assertTrue(np.all(np.isin(X, np.concatenate((x1, x2), axis=0))))
        self.assertEqual(x1.shape[0] + x2.shape[0], X.shape[0])
        self.assertEqual(x1.shape[1], X.shape[1])
        self.assertEqual(x2.shape[1], X.shape[1])

        self.assertTrue(np.all(np.isin(y1, y)))
        self.assertTrue(np.all(np.isin(y2, y)))
        self.assertTrue(np.all(np.isin(np.concatenate((y1, y2), axis=0), y)))
        self.assertTrue(np.all(np.isin(y, np.concatenate((y1, y2), axis=0))))
        self.assertEqual(y1.shape[0] + y2.shape[0], y.shape[0])
        self.assertEqual(y1.shape[1], y.shape[1])
        self.assertEqual(y2.shape[1], y.shape[1])

    def test_split_sets_3d_target1d(self):
        """Split 3d numpy arrays into sets."""
        X = np.arange(1000).reshape(10, 5, 20)
        y = np.arange(50).reshape(10, 5, 1)

        x1, y1, x2, y2 = utils._split_sets(X, y, val=0.5)
        self.assertTrue(np.all(np.isin(x1, X)))
        self.assertTrue(np.all(np.isin(x2, X)))
        self.assertTrue(np.all(np.isin(np.concatenate((x1, x2), axis=0), X)))
        self.assertTrue(np.all(np.isin(X, np.concatenate((x1, x2), axis=0))))
        self.assertEqual(x1.shape[0] + x2.shape[0], X.shape[0])
        self.assertEqual(x1.shape[1], X.shape[1])
        self.assertEqual(x2.shape[1], X.shape[1])
        self.assertEqual(x1.shape[2], X.shape[2])
        self.assertEqual(x2.shape[2], X.shape[2])

        self.assertTrue(np.all(np.isin(y1, y)))
        self.assertTrue(np.all(np.isin(y2, y)))
        self.assertTrue(np.all(np.isin(np.concatenate((y1, y2), axis=0), y)))
        self.assertTrue(np.all(np.isin(y, np.concatenate((y1, y2), axis=0))))
        self.assertEqual(y1.shape[0] + y2.shape[0], y.shape[0])
        self.assertEqual(y1.shape[1], y.shape[1])
        self.assertEqual(y2.shape[1], y.shape[1])
        self.assertEqual(y1.shape[2], y.shape[2])
        self.assertEqual(y2.shape[2], y.shape[2])

    def test_split_sets_2d_target2d(self):
        """Split numpy arrays into sets."""
        X = np.arange(200).reshape(10, 20)
        y = np.arange(20).reshape(10, 2)

        x1, y1, x2, y2 = utils._split_sets(X, y, val=0.5)
        self.assertTrue(np.all(np.isin(x1, X)))
        self.assertTrue(np.all(np.isin(x2, X)))
        self.assertTrue(np.all(np.isin(np.concatenate((x1, x2), axis=0), X)))
        self.assertTrue(np.all(np.isin(X, np.concatenate((x1, x2), axis=0))))
        self.assertEqual(x1.shape[0] + x2.shape[0], X.shape[0])
        self.assertEqual(x1.shape[1], X.shape[1])
        self.assertEqual(x2.shape[1], X.shape[1])

        self.assertTrue(np.all(np.isin(y1, y)))
        self.assertTrue(np.all(np.isin(y2, y)))
        self.assertTrue(np.all(np.isin(np.concatenate((y1, y2), axis=0), y)))
        self.assertTrue(np.all(np.isin(y, np.concatenate((y1, y2), axis=0))))
        self.assertEqual(y1.shape[0] + y2.shape[0], y.shape[0])
        self.assertEqual(y1.shape[1], y.shape[1])
        self.assertEqual(y2.shape[1], y.shape[1])

    def test_split_sets_3d_target2d(self):
        """Split 3d numpy arrays into sets."""
        X = np.arange(1000).reshape(10, 5, 20)
        y = np.arange(100).reshape(10, 5, 2)

        x1, y1, x2, y2 = utils._split_sets(X, y, val=0.5)
        self.assertTrue(np.all(np.isin(x1, X)))
        self.assertTrue(np.all(np.isin(x2, X)))
        self.assertTrue(np.all(np.isin(np.concatenate((x1, x2), axis=0), X)))
        self.assertTrue(np.all(np.isin(X, np.concatenate((x1, x2), axis=0))))
        self.assertEqual(x1.shape[0] + x2.shape[0], X.shape[0])
        self.assertEqual(x1.shape[1], X.shape[1])
        self.assertEqual(x2.shape[1], X.shape[1])
        self.assertEqual(x1.shape[2], X.shape[2])
        self.assertEqual(x2.shape[2], X.shape[2])

        self.assertTrue(np.all(np.isin(y1, y)))
        self.assertTrue(np.all(np.isin(y2, y)))
        self.assertTrue(np.all(np.isin(np.concatenate((y1, y2), axis=0), y)))
        self.assertTrue(np.all(np.isin(y, np.concatenate((y1, y2), axis=0))))
        self.assertEqual(y1.shape[0] + y2.shape[0], y.shape[0])
        self.assertEqual(y1.shape[1], y.shape[1])
        self.assertEqual(y2.shape[1], y.shape[1])
        self.assertEqual(y1.shape[2], y.shape[2])
        self.assertEqual(y2.shape[2], y.shape[2])


class TestSegment(unittest.TestCase):
    """Unit test class for utils._segment function."""

    def test_segment(self):
        X = np.arange(1000).reshape(2, 1, 500)

        # No options check
        x = utils._segment(X)
        np.testing.assert_equal(x, np.stack([X[0, :, 0:200], X[0, :, 200:400],
                                             X[1, :, 0:200], X[1, :, 200:400]]
                                            ))

    def test_segment_sl(self):
        X = np.arange(1000).reshape(2, 1, 500)
        f = np.arange(0, 10)

        # small segment length
        x = utils._segment(X, segment_length=10, augment=False)
        X_true = np.stack([f+ii for ii in range(0, 1000, 10)], axis=0)
        X_true = np.expand_dims(X_true, axis=1)
        np.testing.assert_equal(x, X_true)

    def test_segment_aug_def(self):
        X = np.arange(1000).reshape(2, 1, 500)
        f = np.arange(0, 10)

        # small segment length and augment=true with default value
        x = utils._segment(X, segment_length=10, augment=True)
        X_true = np.stack([f+ii for ii in range(0, 1000, 25)], axis=0)
        X_true = np.expand_dims(X_true, axis=1)
        np.testing.assert_equal(x, X_true)

    def test_segment_aug_custom(self):
        X = np.arange(1000).reshape(2, 1, 500)
        f = np.arange(0, 10)
        # small segment length and augment=true with custom value
        x = utils._segment(X, segment_length=10, augment=True, stride=6)
        X_true = np.vstack(([f+ii for ii in range(0, 490, 6)],
                            [f+ii for ii in range(500, 990, 6)]))
        X_true = np.expand_dims(X_true, axis=1)
        np.testing.assert_equal(x, X_true)

    def test_segment_seq(self):
        X = np.arange(1000).reshape(2, 1, 500)

        # No options check
        x = utils._segment(X, input_type='seq')
        X_t1 = np.array([X[0, :, 0:200], X[0, :, 200:400]])
        X_t2 = np.array([X[1, :, 0:200], X[1, :, 200:400]])
        X_true = np.array([X_t1, X_t2])
        np.testing.assert_equal(x, X_true)

    def test_segment_seq_sl(self):
        X = np.arange(1000).reshape(2, 1, 500)
        f = np.arange(0, 10)

        # small segment length
        x = utils._segment(X, input_type='seq', segment_length=10, augment=False)
        X_t1 = np.stack([f+ii for ii in range(0, 500, 10)], axis=0)
        X_t1 = np.expand_dims(X_t1, axis=1)
        X_t2 = np.stack([f+ii for ii in range(500, 1000, 10)], axis=0)
        X_t2 = np.expand_dims(X_t2, axis=1)
        X_true = np.array([X_t1, X_t2])
        np.testing.assert_equal(x, X_true)

    def test_segment_seq_aug_def(self):
        X = np.arange(1000).reshape(2, 1, 500)
        f = np.arange(0, 10)
        # small segment length and augment=true with default value
        x = utils._segment(X, input_type='seq', segment_length=10, augment=True)
        X_t1 = np.stack([f+ii for ii in range(0, 500, 25)], axis=0)
        X_t1 = np.expand_dims(X_t1, axis=1)
        X_t2 = np.stack([f+ii for ii in range(500, 1000, 25)], axis=0)
        X_t2 = np.expand_dims(X_t2, axis=1)
        X_true = np.array([X_t1, X_t2])
        np.testing.assert_equal(x, X_true)

    def test_segment_seq_aug_custom(self):
        X = np.arange(1000).reshape(2, 1, 500)
        f = np.arange(0, 10)
        # small segment length and augment=true with custom value
        x = utils._segment(X, input_type='seq', segment_length=10,
                           augment=True, stride=6)
        X_t1 = np.stack([f+ii for ii in range(0, 490, 6)], axis=0)
        X_t1 = np.expand_dims(X_t1, axis=1)

        X_t2 = np.stack([f+ii for ii in range(500, 990, 6)], axis=0)
        X_t2 = np.expand_dims(X_t2, axis=1)
        X_true = np.array([X_t1, X_t2])
        np.testing.assert_equal(x, X_true)


class TestMakeExampleProto(unittest.TestCase):
    """Unit test class for utils._make_example function."""

    def test_make_example_trial_int(self):
        """Unit test using numpy arrays, `trial` input and `int` targets."""
        X = np.arange(100).reshape(10, 10)
        y = np.arange(10).reshape(10, 1)
        kk = dict()

        kk['X'] = tf.io.FixedLenFeature([10, 10], tf.float32)
        kk['y'] = tf.io.FixedLenFeature([10, 1], tf.int64)
        example = utils._make_example(X, y,
                                      input_type='trial', target_type='int')
        self.assertTrue(isinstance(example, tf.train.Example))
        self.assertTrue(
            isinstance(example.features.feature['X'], tf.train.Feature))
        self.assertTrue(
            isinstance(example.features.feature['y'], tf.train.Feature))
        a = tf.io.parse_single_example(example.SerializeToString(), kk)

        self.assertTrue(a['X'].dtype.is_floating)
        self.assertTrue(a['y'].dtype.is_integer)

        with tf.compat.v1.Session() as sess:
            a = sess.run(a)
        np.testing.assert_equal(a['X'], X)
        np.testing.assert_equal(a['y'], y)

    def test_make_example_trial_float(self):
        """Unit test using numpy arrays, `trial` input and `float` targets."""
        X = np.arange(100).reshape(10, 10)
        y = np.arange(10).reshape(10, 1)
        kk = dict()

        kk['X'] = tf.io.FixedLenFeature([10, 10], tf.float32)
        kk['y'] = tf.io.FixedLenFeature([10, 1], tf.float32)
        example = utils._make_example(X, y,
                                      input_type='trial', target_type='float')
        self.assertTrue(isinstance(example, tf.train.Example))
        self.assertTrue(
            isinstance(example.features.feature['X'], tf.train.Feature))
        self.assertTrue(
            isinstance(example.features.feature['y'], tf.train.Feature))
        a = tf.io.parse_single_example(example.SerializeToString(), kk)

        self.assertTrue(a['X'].dtype.is_floating)
        self.assertTrue(a['y'].dtype.is_floating)

        with tf.compat.v1.Session() as sess:
            a = sess.run(a)
        np.testing.assert_equal(a['X'], X)
        np.testing.assert_equal(a['y'], y)

    def test_make_example_seq_int(self):
        """Unit test using numpy arrays, `seq` input and `int` targets."""
        X = np.arange(1000).reshape(10, 5, 20)
        y = np.arange(10).reshape(10, 1)
        kk = dict()

        kk['X'] = tf.io.FixedLenFeature([10, 5, 20], tf.float32)
        kk['y'] = tf.io.FixedLenFeature([10, 1], tf.int64)
        example = utils._make_example(X, y,
                                      input_type='seq', target_type='int')
        self.assertTrue(isinstance(example, tf.train.Example))
        self.assertTrue(
            isinstance(example.features.feature['X'], tf.train.Feature))
        self.assertTrue(
            isinstance(example.features.feature['y'], tf.train.Feature))
        a = tf.io.parse_single_example(example.SerializeToString(), kk)

        self.assertTrue(a['X'].dtype.is_floating)
        self.assertTrue(a['y'].dtype.is_integer)

        with tf.compat.v1.Session() as sess:
            a = sess.run(a)
        np.testing.assert_equal(a['X'], X)
        np.testing.assert_equal(a['y'], y)

    def test_make_example_seq_float(self):
        """Unit test using numpy arrays, `seq` input and `float` targets."""
        X = np.arange(1000).reshape(10, 5, 20)
        y = np.arange(10).reshape(10, 1)
        kk = dict()

        kk['X'] = tf.io.FixedLenFeature([10, 5, 20], tf.float32)
        kk['y'] = tf.io.FixedLenFeature([10, 1], tf.float32)
        example = utils._make_example(X, y,
                                      input_type='seq', target_type='float')
        self.assertTrue(isinstance(example, tf.train.Example))
        self.assertTrue(
            isinstance(example.features.feature['X'], tf.train.Feature))
        self.assertTrue(
            isinstance(example.features.feature['y'], tf.train.Feature))
        a = tf.io.parse_single_example(example.SerializeToString(), kk)

        self.assertTrue(a['X'].dtype.is_floating)
        self.assertTrue(a['y'].dtype.is_floating)

        with tf.compat.v1.Session() as sess:
            a = sess.run(a)
        np.testing.assert_equal(a['X'], X)
        np.testing.assert_equal(a['y'], y)


class TestWriteTFRecords(unittest.TestCase):
    """Unit test class for utils._write_tfrecords function."""

    def test_write_tfrecords_trial_int(self):
        """Unit test using numpy arrays, `trial` input and `int` targets."""
        X_ = np.arange(1000).reshape(10, 5, 20)
        y_ = np.arange(10).reshape(10, 1)
        savepath = './tmptfr/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fname = savepath+time.strftime('%Y%m%d_%H%M%S_')+'example.tfrecord'

        def _parse_tfrecords_trial_int(example_proto):
            kk = dict()
            kk['X'] = tf.io.FixedLenFeature([5, 20], tf.float32)
            kk['y'] = tf.io.FixedLenFeature([1], tf.int64)
            return tf.io.parse_single_example(example_proto, kk)

        utils._write_tfrecords(X_, y_, fname,
                               input_type='trials', target_type='int')
        a = tf.data.TFRecordDataset(fname)
        a = a.map(_parse_tfrecords_trial_int)
        iterator = tf.compat.v1.data.make_one_shot_iterator(a)
        next_element = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            try:
                for ii in range(10):
                    data = sess.run(next_element)
                    np.testing.assert_equal(X_[ii], data['X'])
                    np.testing.assert_equal(y_[ii], data['y'])
            except Exception as e:
                print("Unexpected error:", type(e), e)
                raise
            finally:
                os.remove(fname)

    def test_write_tfrecords_trial_float(self):
        """Unit test using numpy arrays, `trial` input and `float` targets."""
        X_ = np.arange(1000).reshape(10, 5, 20)
        y_ = np.arange(10).reshape(10, 1)
        savepath = './tmptfr/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fname = savepath+time.strftime('%Y%m%d_%H%M%S_')+'example.tfrecord'

        def _parse_tfrecords_trial_float(example_proto):
            kk = dict()
            kk['X'] = tf.io.FixedLenFeature([5, 20], tf.float32)
            kk['y'] = tf.io.FixedLenFeature([1], tf.float32)
            return tf.io.parse_single_example(example_proto, kk)

        utils._write_tfrecords(X_, y_, fname,
                               input_type='iid', target_type='float')
        a = tf.data.TFRecordDataset(fname)
        a = a.map(_parse_tfrecords_trial_float)
        iterator = tf.compat.v1.data.make_one_shot_iterator(a)
        next_element = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            try:
                for ii in range(10):
                    data = sess.run(next_element)
                    np.testing.assert_equal(X_[ii], data['X'])
                    np.testing.assert_equal(y_[ii], data['y'])
            except Exception as e:
                print("Unexpected error:", type(e), e)
                raise
            finally:
                os.remove(fname)

    def test_write_tfrecords_seq_int(self):
        """Unit test using numpy arrays, `seq` input and `int` targets."""
        X_ = np.arange(1000).reshape(10, 5, 20)
        y_ = np.arange(10).reshape(10, 1)
        savepath = './tmptfr/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fname = savepath+time.strftime('%Y%m%d_%H%M%S_')+'example.tfrecord'

        def _parse_tfrecords_seq_int(example_proto):
            kk = dict()
            kk['X'] = tf.io.FixedLenFeature([5, 20], tf.float32)
            kk['y'] = tf.io.FixedLenFeature([1], tf.int64)
            return tf.io.parse_single_example(example_proto, kk)

        utils._write_tfrecords(X_, y_, fname,
                               input_type='seq', target_type='int')
        a = tf.data.TFRecordDataset(fname)
        a = a.map(_parse_tfrecords_seq_int)
        iterator = tf.compat.v1.data.make_one_shot_iterator(a)
        next_element = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            try:
                for ii in range(10):
                    data = sess.run(next_element)
                    np.testing.assert_equal(X_[ii], data['X'])
                    np.testing.assert_equal(y_[ii], data['y'])
            except Exception as e:
                print("Unexpected error:", type(e), e)
                raise
            finally:
                os.remove(fname)

    def test_write_tfrecords_seq_float(self):
        """Unit test using numpy arrays, `seq` input and `float` targets."""
        X_ = np.arange(1000).reshape(10, 5, 20)
        y_ = np.arange(10).reshape(10, 1)
        savepath = './tmptfr/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fname = savepath+time.strftime('%Y%m%d_%H%M%S_')+'example.tfrecord'

        def _parse_tfrecords_seq_float(example_proto):
            kk = dict()
            kk['X'] = tf.io.FixedLenFeature([5, 20], tf.float32)
            kk['y'] = tf.io.FixedLenFeature([1], tf.float32)
            return tf.io.parse_single_example(example_proto, kk)

        utils._write_tfrecords(X_, y_, fname,
                               input_type='seq', target_type='float')
        a = tf.data.TFRecordDataset(fname)
        a = a.map(_parse_tfrecords_seq_float)
        iterator = tf.compat.v1.data.make_one_shot_iterator(a)
        next_element = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            try:
                for ii in range(10):
                    data = sess.run(next_element)
                    np.testing.assert_equal(X_[ii], data['X'])
                    np.testing.assert_equal(y_[ii], data['y'])
            except Exception as e:
                print("Unexpected error:", type(e), e)
                raise
            finally:
                os.remove(fname)


class TestImportData(unittest.TestCase):
    """Unit test class for utils.import_data function."""

    def test_import_data_epochs(self):
        """Test data import using a mne.Epochs file."""
        epochs = mne.read_epochs('example-epo.fif', preload=True)
        picks = mne.pick_types(epochs.info, meg='grad')
        # picks = np.arange(0, 204)

        # All channels
        data, events = utils.import_data(epochs)
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 316, 361), data.shape)
        self.assertEqual((221, 1), events.shape)

        # Pick channels
        data, events = utils.import_data(epochs, picks=picks)
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 204, 361), data.shape)
        self.assertEqual((221, 1), events.shape)

        # Transpose data
        data, events = utils.import_data(epochs, transpose=['X', 'y'])
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 361, 316), data.shape)
        self.assertEqual((221, 1), events.shape)

    def test_import_data_tuple(self):
        """Test data import using a tuple."""
        X_ = np.arange(1000).reshape(10, 5, 20)
        y_ = np.arange(20).reshape(10, 2)
        picks = [2, 4]

        # All channels
        data, events = utils.import_data((X_, y_))
        self.assertEqual(data.ndim, 3)
        np.testing.assert_array_equal(X_, data)
        np.testing.assert_array_equal(y_, events)

        # Pick channels
        data, events = utils.import_data((X_, y_), picks=picks)
        self.assertEqual(data.ndim, 3)
        np.testing.assert_array_equal(X_[:, picks, :], data)
        np.testing.assert_array_equal(y_, events)

        # Transpose data
        data, events = utils.import_data((X_, y_), transpose=['X', 'y'])
        self.assertEqual(data.ndim, 3)
        self.assertEqual((10, 20, 5), data.shape)
        self.assertEqual((10, 2), events.shape)

    def test_import_data_fif(self):
        """Test data import using the name of a .fif file."""
        picks = np.arange(0, 204)

        # All channels
        data, events = utils.import_data('example-epo.fif')
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 316, 361), data.shape)
        self.assertEqual((221, 1), events.shape)

        # Pick channels
        data, events = utils.import_data('example-epo.fif', picks=picks)
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 204, 361), data.shape)
        self.assertEqual((221, 1), events.shape)

        # Transpose data
        data, events = utils.import_data('example-epo.fif',
                                         transpose=['X', 'y'])
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 361, 316), data.shape)
        self.assertEqual((221, 1), events.shape)

    def test_import_data_npz(self):
        """Test data import using the name of a .npz file."""
        kk = {'X': 'data', 'y': 'events'}
        picks = np.arange(0, 102)

        # All channels
        data, events = utils.import_data('example_meg.npz', array_keys=kk)
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 316, 361), data.shape)
        self.assertEqual((221, 1), events.shape)

        # Pick channels
        data, events = utils.import_data('example_meg.npz', array_keys=kk,
                                         picks=picks)
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 102, 361), data.shape)
        self.assertEqual((221, 1), events.shape)

        # Transpose data
        data, events = utils.import_data('example_meg.npz', array_keys=kk,
                                         transpose=['X', 'y'])
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 361, 316), data.shape)
        self.assertEqual((221, 1), events.shape)

    def test_import_data_mat(self):
        """Test data import using the name of a .mat file."""
        kk = {'X': 'data', 'y': 'events'}
        picks = np.arange(0, 102)

        # All channels
        data, events = utils.import_data('example_meg.mat', array_keys=kk)
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 316, 361), data.shape)
        self.assertEqual((221, 1), events.shape)

        # Pick channels
        data, events = utils.import_data('example_meg.mat', array_keys=kk,
                                         picks=picks)
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 102, 361), data.shape)
        self.assertEqual((221, 1), events.shape)

        # Transpose data
        data, events = utils.import_data('example_meg.mat', array_keys=kk,
                                         transpose=['X', 'y'])
        self.assertEqual(data.ndim, 3)
        self.assertEqual((221, 361, 316), data.shape)
        self.assertEqual((221, 1), events.shape)


class TestProduceTFRecords(unittest.TestCase):
    """Unit test class for utils.produce_tfrecords function."""

    def test_produce_tfrecords(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        # no optional inputs
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_trial_reg()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            print(ii)
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    # ---- TRIALS ----
    # -- classification ---
    def test_produce_tfrecords_trials_int_holdout(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='trials',
            target_type='int',
            test_set='holdout',
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_trial_class()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    def test_produce_tfrecords_trials_int_loso(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='trials',
            target_type='int',
            test_set='loso',
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_trial_class()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221*2)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    def test_produce_tfrecords_trials_int_none(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='trials',
            target_type='int',
            test_set=None,
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_trial_class()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    # --- reg ---
    def test_produce_tfrecords_trials_float_holdout(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='trials',
            target_type='float',
            test_set='holdout',
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_trial_reg()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    def test_produce_tfrecords_trials_float_loso(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='trials',
            target_type='float',
            test_set='loso',
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_trial_reg()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221*2)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    def test_produce_tfrecords_trials_float_none(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='trials',
            target_type='float',
            test_set=None,
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_trial_reg()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    # --- SEQ ---
    # -- int ---
    def test_produce_tfrecords_seq_int_holdout(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='seq',
            target_type='int',
            test_set='holdout',
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_seq_class()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    def test_produce_tfrecords_seq_int_loso(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='seq',
            target_type='int',
            test_set='loso',
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_seq_class()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221*2)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    def test_produce_tfrecords_seq_int_none(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='seq',
            target_type='int',
            test_set=None,
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_seq_class()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    # --- reg ---
    def test_produce_tfrecords_seq_float_holdout(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='seq',
            target_type='float',
            test_set='holdout',
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_seq_reg()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    def test_produce_tfrecords_seq_float_loso(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='seq',
            target_type='float',
            test_set='loso',
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_seq_reg()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221*2)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)

    def test_produce_tfrecords_seq_float_none(self):
        inp_fname = 'example-epo.fif'
        savepath = './tmptfr/'
        out_name = time.strftime('%Y%m%d_%H%M%S_')+'example'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        opts = dict(
            input_type='seq',
            target_type='float',
            test_set=None,
            overwrite=True,
            fs=600)

        # trials int holdout
        meta = utils.produce_tfrecords(inp_fname, savepath, out_name, **opts)
        s = utils._load_meta(savepath+out_name)
        self.assertEqual(meta, s)
        tmp = check_meta_seq_reg()
        self.assertEqual(set(s.keys()), set(tmp.keys()))
        for ii in s.keys():
            self.assertEqual(type(tmp[ii]), type(s[ii]))
        os.remove(savepath+out_name+'_meta.pkl')

        t, e = check_tfrecords(path=savepath, pattern=out_name)
        self.assertEqual(t, 221)
        self.assertEqual(e, 0)
        for f in sorted(glob.glob('%s*%s*.tfrecord' % (savepath, out_name))):
            os.remove(f)


class TestFinalCleanup(unittest.TestCase):
    """Clean up after all unit tests."""

    def test_cleanup(self):
        import os
        import shutil

        for root, dirs, files in os.walk('./tfr/'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))


# --- MAIN ---
if __name__ == '__main__':
    # unittest.main()
    print('Loaded')
