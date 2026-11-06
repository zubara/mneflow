"""Tests for the connectivity models in mneflow.fc_models.

Uses small synthetic connectivity data (6 nodes, 4 frequencies) written with
produce_tfrecords, so the models are built, trained (with cross-validation
and a holdout set), logged and saved through the real pipeline.
"""
import contextlib
import csv
import io
import os

import keras
import numpy as np
import pytest

import mneflow
from mneflow.fc_models import (Conv3DModel, WeightedSum3dModel,
                               SymmetricModel, _fold_stability)

N_SAMPLES, N_NODES, N_FREQS, N_FOLDS = 96, 6, 4, 3

# (model class, model specs, expected scope)
CONFIGS = {
    'conv3d': (Conv3DModel, {'n_latent': 4}, 'conv3d'),
    'weighted_sum': (WeightedSum3dModel, {'n_latent1': 4, 'n_latent2': 2},
                     'weighted_sum_3d'),
    'symmetric': (SymmetricModel, {'n_latent': 4,
                                   'trainable_pointwise_kernel': True},
                  'symmetric_tpk'),
    'symmetric_fixed_pointwise': (SymmetricModel,
                                  {'n_latent': 4,
                                   'trainable_pointwise_kernel': False},
                                  'symmetric_tpk'),
}
COMMON_SPECS = dict(n_latent_dense=8, dropout=0.1, stddev=0.0,
                    spatial_dropout=0.1, l1_lambda=0., l2_lambda=0.,
                    l1_scope=[], l2_scope=[], unitnorm_scope=[])


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def make_data(signal=0.8, seed=0):
    """Symmetric connectivity whose level encodes a sorted target."""
    rng = np.random.default_rng(seed)
    y = np.sort(rng.uniform(20, 80, N_SAMPLES))
    level = (y - 20) / 60
    X = (rng.random((N_SAMPLES, N_NODES, N_NODES, N_FREQS)) * (1 - signal)
         + signal * level[:, None, None, None]).astype(np.float32)
    X = (X + X.transpose(0, 2, 1, 3)) / 2
    X[:, np.arange(N_NODES), np.arange(N_NODES), :] = 0
    return X, y


@pytest.fixture(scope='module')
def tfr_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('fc_tfr')
    np.random.seed(0)
    _quiet(mneflow.produce_tfrecords, make_data(), path=str(path),
           data_id='fc', fs=1, input_type='fconn', target_type='float',
           n_folds=N_FOLDS, overwrite=True, test_set='holdout')
    return path


def make_model(tfr_path, name, model_id=None, **extra_specs):
    cls, specs, _ = CONFIGS[name]
    keras.utils.set_random_seed(0)
    meta = mneflow.load_meta(str(tfr_path), 'fc')
    # Training saves the model specs into the shared metadata file, so
    # start every model from empty specs
    meta.model_specs = {}
    meta.update(model_specs=dict(COMMON_SPECS, **specs, **extra_specs,
                                 model_id=model_id or name))
    dataset = mneflow.Dataset(meta, train_batch=16)
    model = _quiet(cls, meta=meta, dataset=dataset, specs_prefix=False)
    _quiet(model.build, loss='MAE', metrics='RootMeanSquaredError',
           learn_rate=1e-2)
    return model


def train_cv(model, n_epochs=2, eval_step=3, early_stopping=1):
    _quiet(model.train, n_epochs=n_epochs, eval_step=eval_step,
           early_stopping=early_stopping, mode='cv',
           collect_patterns=isinstance(model, SymmetricModel))


def read_log(model):
    path = os.path.join(model.model_path, model.scope + '_log.csv')
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_list(text):
    return [float(x) for x in
            text.replace('np.float64(', '').replace(')', '').strip('[]')
            .split(',') if x.strip()]


@pytest.mark.parametrize('name', list(CONFIGS))
def test_builds_with_expected_output_shape(tfr_path, name):
    model = make_model(tfr_path, name)
    assert model.y_pred.shape == (None, 1)


@pytest.mark.parametrize('name', list(CONFIGS))
def test_scope_is_consistent(tfr_path, name):
    """The scope names the log and saved model files, and is stored in
    the metadata; all of these must agree."""
    model = make_model(tfr_path, name)
    assert model.scope == CONFIGS[name][2]
    assert model.meta.model_specs['scope'] == model.scope
    assert model.model_name.startswith(model.scope + '_')


def test_symmetric_model_uses_attention_rate(tfr_path):
    model = make_model(tfr_path, 'symmetric', attention_rate=2)
    assert model.specs['attention_rate'] == 2
    assert 'attention_ratio' not in model.specs


@pytest.mark.parametrize('name', list(CONFIGS))
def test_cv_training_writes_log_and_saves_model(tfr_path, name):
    model = make_model(tfr_path, name, model_id=f'log_test_{name}')
    train_cv(model)
    rows = [r for r in read_log(model)
            if r['spec_model_id'] == f'log_test_{name}']
    assert len(rows) == 1
    row = rows[0]
    assert row['loss'] == 'MAE'
    assert row['model_id'] == model.model_name
    # One validation and one holdout score per CV fold
    assert len(parse_list(row['cv_losses'])) == N_FOLDS
    assert len(parse_list(row['test_losses'])) == N_FOLDS
    assert all(np.isfinite(parse_list(row['test_losses'])))
    if name == 'weighted_sum':
        assert row['n_latent'] == ''  # this model has no n_latent
    assert os.path.exists(os.path.join(model.model_path,
                                       model.model_name + '.h5'))


def test_symmetric_patterns_and_roi_labels(tfr_path):
    model = make_model(tfr_path, 'symmetric')
    train_cv(model)
    # One collected pattern per CV fold (the holdout fold is not trained)
    dcov = model.cv_patterns['dcov']
    assert dcov.shape[-1] == N_FOLDS
    assert all(np.any(dcov[..., i]) for i in range(N_FOLDS))
    patterns = _quiet(model.extract_patterns)
    for key in ('row', 'col', 'diag'):
        assert patterns['spatial_patterns'][key].shape == (N_NODES, N_FOLDS)

    # Default return value is unchanged: a dict of label indices
    roi = _quiet(model.get_roi_labels, patterns=patterns)
    assert set(roi) == {'row', 'col', 'diag', 'row-col'}

    roi2, stability = _quiet(model.get_roi_labels, patterns=patterns,
                             return_stability=True)
    assert set(roi2) == set(roi)
    for key in ('row', 'col', 'diag'):
        assert 0 <= stability[key]['jaccard'] <= 1
        assert 0 <= stability[key]['abs_r'] <= 1


@pytest.mark.xfail(raises=IndexError, strict=True,
                   reason="extract_patterns assumes the trainable pointwise "
                          "layer's weight shape; with "
                          "trainable_pointwise_kernel=False the 1x1 Conv2D "
                          "kernel has one dimension fewer")
def test_symmetric_patterns_without_trainable_pointwise(tfr_path):
    model = make_model(tfr_path, 'symmetric_fixed_pointwise')
    train_cv(model)
    _quiet(model.extract_patterns)


def test_fold_stability_measures():
    rng = np.random.default_rng(0)
    pattern = rng.normal(size=(50, 1))
    same = np.repeat(pattern, 3, axis=1)
    stab = _fold_stability(same, threshold=1.)
    assert stab['jaccard'] == pytest.approx(1)
    assert stab['abs_r'] == pytest.approx(1)

    unrelated = rng.normal(size=(5000, 3))
    stab = _fold_stability(unrelated, threshold=1.)
    assert stab['abs_r'] < 0.1
    assert stab['jaccard'] < 0.3


@pytest.mark.slow
@pytest.mark.parametrize('name', ['conv3d', 'weighted_sum', 'symmetric'])
def test_models_learn_target_signal(tfr_path, name):
    """On data whose connectivity level encodes the target, the holdout
    error must be clearly below predicting the mean training target
    (a model stuck at chance level scores about the same as that)."""
    model = make_model(tfr_path, name, model_id=f'learn_{name}')
    train_cv(model, n_epochs=60, eval_step=10, early_stopping=10)
    row = [r for r in read_log(model)
           if r['spec_model_id'] == f'learn_{name}'][0]
    holdout_mae = np.mean(parse_list(row['test_losses']))

    _, y = make_data()
    test_fold = np.asarray(model.meta.data['test_fold'])
    train_y = np.delete(y, test_fold)
    baseline_mae = np.mean(np.abs(y[test_fold] - train_y.mean()))
    assert holdout_mae < 0.7 * baseline_mae
