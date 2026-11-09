"""Tests for how produce_tfrecords splits data into folds and writes records.

The synthetic samples carry their own index: every value of sample i is
i + 1, and the targets are sorted (like subjects ordered by age). This makes
it easy to check that each record's data, target and index 'n' belong
together, and that folds, holdout set and records agree.
"""

import contextlib
import io

import numpy as np
import pytest
import tensorflow as tf
from scipy.stats import spearmanr

import mneflow
from mneflow.utils import _split_indices, minmax_scale


def _quiet(func, *args, **kwargs):
    """Call func without its (very verbose) printing."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def make_fconn(n_samples=120, n_ch=4, n_freq=2, start=0):
    X = np.zeros((n_samples, n_ch, n_ch, n_freq), np.float32)
    X[:] = (np.arange(n_samples) + start + 1)[:, None, None, None]
    y = np.arange(n_samples, dtype=float) + start  # sorted targets
    return X, y


def produce(tmp_path, inputs, n_folds=5, test_set='holdout', **kwargs):
    opts = dict(
        path=str(tmp_path),
        data_id='test',
        fs=1,
        input_type='fconn',
        target_type='float',
        n_folds=n_folds,
        overwrite=True,
        test_set=test_set,
    )
    opts.update(kwargs)
    return _quiet(mneflow.produce_tfrecords, inputs, **opts)


def read_records(meta, paths):
    """(n, first X value, first y value) of each record, in file order."""
    parse = mneflow.Dataset(meta)._parse_function
    out = [
        (
            int(r['n']),
            float(r['X'].numpy().ravel()[0]),
            float(r['y'].numpy().ravel()[0]),
        )
        for r in tf.data.TFRecordDataset(paths).map(parse)
    ]
    return np.array(out).reshape(-1, 3)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_split_indices_are_disjoint_and_equal_sized():
    folds = _split_indices(103, n_folds=5)
    assert [len(f) for f in folds] == [20] * 5
    all_inds = np.concatenate(folds)
    assert len(np.unique(all_inds)) == len(all_inds)
    assert all_inds.max() < 103


def test_holdout_split_covers_all_samples(tmp_path):
    X, y = make_fconn(120)
    meta = produce(tmp_path, (X, y))
    folds = meta.data['folds'][0]
    test_fold = np.asarray(meta.data['test_fold'])
    assert len(folds) == 5 and all(len(f) == 20 for f in folds)
    assert len(test_fold) == 20
    all_inds = np.concatenate(folds + [test_fold])
    np.testing.assert_array_equal(np.sort(all_inds), np.arange(120))


def test_records_match_their_sample_index(tmp_path):
    X, y = make_fconn(120)
    meta = produce(tmp_path, (X, y))
    train = read_records(meta, meta.data['train_paths'])
    test = read_records(meta, meta.data['test_paths'])
    for recs in (train, test):
        np.testing.assert_allclose(recs[:, 1], recs[:, 0] + 1)  # X
        np.testing.assert_allclose(recs[:, 2], recs[:, 0])  # y
    # train records = everything except the holdout; test records = holdout
    test_fold = np.asarray(meta.data['test_fold'])
    np.testing.assert_array_equal(np.sort(test[:, 0]), np.sort(test_fold))
    np.testing.assert_array_equal(
        np.sort(train[:, 0]), np.setdiff1d(np.arange(120), test_fold)
    )


def test_train_records_are_shuffled(tmp_path):
    X, y = make_fconn(120)
    meta = produce(tmp_path, (X, y))
    train = read_records(meta, meta.data['train_paths'])
    # Inputs are sorted by target; the written order must not be
    rho = spearmanr(np.arange(len(train)), train[:, 2]).statistic
    assert abs(rho) < 0.3


@pytest.mark.parametrize('test_set', ['holdout', None])
def test_dataset_cv_filters_select_intended_folds(tmp_path, test_set):
    X, y = make_fconn(120)
    meta = produce(tmp_path, (X, y), test_set=test_set)
    folds = meta.data['folds'][0]
    n_train = 120 - (20 if test_set == 'holdout' else 0)
    dataset = mneflow.Dataset(meta)
    for k in range(len(folds)):
        _quiet(
            dataset._build_dataset,
            meta.data['train_paths'],
            split=True,
            val_fold_ind=k,
        )
        records = tf.data.TFRecordDataset(meta.data['train_paths']).map(
            dataset._parse_function
        )
        train_n = {
            int(r['n']) for r in records.filter(dataset._cv_train_fold_filter)
        }
        val_n = {
            int(r['n']) for r in records.filter(dataset._cv_val_fold_filter)
        }
        assert val_n == set(folds[k])
        others = np.concatenate([f for i, f in enumerate(folds) if i != k])
        assert train_n == set(others)
        assert len(train_n) + len(val_n) == n_train
        assert not train_n & val_n


def test_predefined_split_is_used_exactly(tmp_path):
    X, y = make_fconn(120)
    # Contiguous blocks in reverse order, last block = holdout
    split = [[np.arange(i * 20, (i + 1) * 20) for i in reversed(range(6))]]
    split_copy = [[f.copy() for f in split[0]]]
    meta = produce(tmp_path, (X, y), predefined_split=split)

    for got, expected in zip(meta.data['folds'][0], split_copy[0][:-1]):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(meta.data['test_fold'], split_copy[0][-1])
    test = read_records(meta, meta.data['test_paths'])
    np.testing.assert_array_equal(np.sort(test[:, 0]), np.arange(0, 20))
    # The caller's split is not modified, so it can be reused
    assert len(split[0]) == 6
    meta2 = produce(tmp_path, (X, y), predefined_split=split, data_id='test2')
    np.testing.assert_array_equal(meta2.data['test_fold'], split_copy[0][-1])


def test_predefined_split_with_wrong_sizes_raises(tmp_path):
    X, y = make_fconn(120)
    too_few = [[np.arange(i * 20, (i + 1) * 20) for i in range(5)]]
    with pytest.raises(AssertionError):
        produce(tmp_path, (X, y), predefined_split=too_few)
    uneven = [
        [np.arange(0, 30)]
        + [np.arange(30 + i * 18, 48 + i * 18) for i in range(5)]
    ]
    with pytest.raises(AssertionError):
        produce(tmp_path, (X, y), predefined_split=uneven)


def test_multiple_inputs_get_unique_indices(tmp_path):
    inputs = [make_fconn(60, start=0), make_fconn(60, start=60)]
    meta = produce(tmp_path, inputs)
    train = read_records(meta, meta.data['train_paths'])
    test = read_records(meta, meta.data['test_paths'])
    all_n = np.concatenate([train[:, 0], test[:, 0]])
    np.testing.assert_array_equal(np.sort(all_n), np.arange(120))
    np.testing.assert_allclose(train[:, 1], train[:, 0] + 1)
    # The second input's folds refer to its own samples (60..119)
    second = np.concatenate(meta.data['folds'][1])
    assert second.min() >= 60 and second.max() < 120


def test_multiple_inputs_holdout_test_fold_covers_all_test_files(tmp_path):
    """With several inputs, test_fold lists the holdout samples of all
    inputs, in the order of the test files."""
    inputs = [make_fconn(60, start=0), make_fconn(60, start=60)]
    meta = produce(tmp_path, inputs)
    test = read_records(meta, meta.data['test_paths'])
    np.testing.assert_array_equal(test[:, 0], meta.data['test_fold'])
    assert len(meta.data['test_fold']) == meta.data['test_size']
    # Both inputs contribute holdout samples
    assert np.any(test[:, 0] < 60) and np.any(test[:, 0] >= 60)


def test_loso_writes_whole_inputs_as_test_sets(tmp_path):
    inputs = [make_fconn(60, start=0), make_fconn(60, start=60)]
    meta = produce(tmp_path, inputs, test_set='loso')
    assert len(meta.data['train_paths']) == 2
    for i, path in enumerate(meta.data['test_paths']):
        recs = read_records(meta, [path])
        np.testing.assert_array_equal(
            np.sort(recs[:, 0]), np.arange(60) + 60 * i
        )
        np.testing.assert_allclose(recs[:, 1], recs[:, 0] + 1)


def test_trials_input_with_holdout(tmp_path):
    n, n_ch, n_t = 100, 3, 16
    X = np.zeros((n, n_ch, n_t), np.float32)
    X[:] = (np.arange(n) + 1)[:, None, None]
    y = np.repeat([0, 1], n // 2)  # sorted classes
    meta = produce(tmp_path, (X, y), input_type='trials', target_type='int')
    parse = mneflow.Dataset(meta)._parse_function
    recs = [
        (
            int(r['n']),
            float(r['X'].numpy().ravel()[0]),
            int(np.argmax(r['y'].numpy())),
        )
        for r in tf.data.TFRecordDataset(meta.data['train_paths']).map(parse)
    ]
    recs = np.array(recs)
    np.testing.assert_allclose(recs[:, 1], recs[:, 0] + 1)
    np.testing.assert_array_equal(recs[:, 2], y[recs[:, 0].astype(int)])
    assert len(recs) == n - len(meta.data['test_fold'])


def test_minmax_scale_keeps_zeros_and_scales_each_sample():
    X = np.array(
        [[[0, 2, 4], [2, 0, 6], [4, 6, 0]], [[0, 5, 5], [5, 0, 5], [5, 5, 0]]],
        float,
    )
    X_orig = X.copy()
    scaled = minmax_scale(X)
    np.testing.assert_allclose(
        scaled[0], [[0, 0, 0.5], [0, 0, 1], [0.5, 1, 0]]
    )
    np.testing.assert_allclose(scaled[1], 0)  # constant sample
    np.testing.assert_array_equal(X, X_orig)  # input not modified


def test_fconn_scale_option_min_max_scales_samples(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.uniform(1, 5, (60, 4, 4, 2)).astype(np.float32)
    y = np.arange(60, dtype=float)
    meta = produce(tmp_path, (X, y), scale=True, test_set=None)
    parse = mneflow.Dataset(meta)._parse_function
    for r in tf.data.TFRecordDataset(meta.data['train_paths']).map(parse):
        x = r['X'].numpy()
        assert x.min() == pytest.approx(0, abs=1e-6)
        assert x.max() == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize('test_set', [None, 'holdout'])
def test_continuous_input(tmp_path, test_set):
    """A continuous recording is cut into segments and split into folds;
    the target signal stays aligned with the data."""
    n_t = 6000
    t = np.arange(n_t, dtype=np.float32)
    data = np.zeros((1, 4, n_t), np.float32)
    data[:] = t  # value = time point
    events = 2 * t[None, None, :]  # target = 2 * time point
    meta = produce(
        tmp_path,
        (data, events),
        input_type='continuous',
        target_type='signal',
        test_set=test_set,
    )
    folds = meta.data['folds'][0]
    assert len(folds) == 5
    parse = mneflow.Dataset(meta)._parse_function
    recs = [
        (int(r['n']), r['X'].numpy(), r['y'].numpy())
        for r in tf.data.TFRecordDataset(meta.data['train_paths']).map(parse)
    ]
    for _, x, y in recs:
        np.testing.assert_allclose(y.ravel()[0], 2 * x.ravel()[0])
    n = {r[0] for r in recs}
    assert set(np.concatenate(folds)) <= n
    if test_set == 'holdout':
        assert not n & set(np.asarray(meta.data['test_fold']))


def test_dataset_timepoint_subset(tmp_path):
    """Dataset keeps only the time points in meta.data['timepoint_subset']."""
    X = np.random.rand(60, 3, 20).astype(np.float32)
    y = np.repeat([0, 1], 30)
    meta = produce(
        tmp_path,
        (X, y),
        input_type='trials',
        target_type='int',
        n_folds=3,
        test_set=None,
    )
    meta.data['timepoint_subset'] = [0, 5, 10]
    dataset = _quiet(mneflow.Dataset, meta, train_batch=10)
    x, _ = next(iter(dataset.train))
    assert x.shape[2] == 3  # (batch, n_seq, times, channels)


def test_missing_input_returns_none(tmp_path):
    """Without data (and no saved metadata), produce_tfrecords reports that
    the dataset was not found and returns None instead of crashing."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        meta = mneflow.produce_tfrecords(
            None,
            path=str(tmp_path),
            data_id='missing',
            fs=1,
            input_type='fconn',
            target_type='float',
            overwrite=False,
        )
    assert meta is None
    assert 'Dataset not found' in out.getvalue()


def test_data_with_wrong_dimensions_raises(tmp_path):
    X = np.random.rand(60, 2, 3, 4, 5).astype(np.float32)  # 5-D
    y = np.arange(60, dtype=float)
    with pytest.raises(ValueError, match='3 or 4 dimensions'):
        produce(tmp_path, (X, y), input_type='trials', test_set=None)


def test_scale_to_baseline_rejects_invalid_baseline():
    from mneflow.utils import scale_to_baseline

    with pytest.raises(ValueError, match='baseline'):
        _quiet(scale_to_baseline, np.random.rand(4, 3, 20), baseline=[0, 5])


def test_dataset_test_set_is_in_test_fold_order(tmp_path):
    """The test set is not shuffled, so its samples come in the order of
    meta.data['test_fold'] (needed to match predictions to samples)."""
    X, y = make_fconn(120)
    meta = produce(tmp_path, (X, y))
    dataset = _quiet(mneflow.Dataset, meta)
    test = _quiet(
        dataset._build_dataset,
        meta.data['test_paths'],
        split=False,
        repeat=False,
    )
    _, y_test = next(iter(test))
    np.testing.assert_allclose(
        np.ravel(y_test), y[np.asarray(meta.data['test_fold'])]
    )
