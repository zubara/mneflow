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
from mneflow.fc_models import (
    Conv3DModel,
    WeightedSum3dModel,
    SymmetricModel,
    _fold_stability,
)

N_SAMPLES, N_NODES, N_FREQS, N_FOLDS = 96, 6, 4, 3

# (model class, model specs, expected scope)
CONFIGS = {
    'conv3d': (Conv3DModel, {'n_latent': 4}, 'conv3d'),
    'weighted_sum': (
        WeightedSum3dModel,
        {'n_latent1': 4, 'n_latent2': 2},
        'weighted_sum_3d',
    ),
    'symmetric': (
        SymmetricModel,
        {'n_latent': 4, 'trainable_pointwise_kernel': True},
        'symmetric_tpk',
    ),
    'symmetric_fixed_pointwise': (
        SymmetricModel,
        {'n_latent': 4, 'trainable_pointwise_kernel': False},
        'symmetric_tpk',
    ),
}
COMMON_SPECS = dict(
    n_latent_dense=8,
    dropout=0.1,
    stddev=0.0,
    spatial_dropout=0.1,
    l1_lambda=0.0,
    l2_lambda=0.0,
    l1_scope=[],
    l2_scope=[],
    unitnorm_scope=[],
)


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def make_data(signal=0.8, seed=0):
    """Symmetric connectivity whose level encodes a sorted target."""
    rng = np.random.default_rng(seed)
    y = np.sort(rng.uniform(20, 80, N_SAMPLES))
    level = (y - 20) / 60
    X = (
        rng.random((N_SAMPLES, N_NODES, N_NODES, N_FREQS)) * (1 - signal)
        + signal * level[:, None, None, None]
    ).astype(np.float32)
    X = (X + X.transpose(0, 2, 1, 3)) / 2
    X[:, np.arange(N_NODES), np.arange(N_NODES), :] = 0
    return X, y


@pytest.fixture(scope='module')
def tfr_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('fc_tfr')
    np.random.seed(0)
    _quiet(
        mneflow.produce_tfrecords,
        make_data(),
        path=str(path),
        data_id='fc',
        fs=1,
        input_type='fconn',
        target_type='float',
        n_folds=N_FOLDS,
        overwrite=True,
        test_set='holdout',
    )
    return path


def make_model(tfr_path, name, model_id=None, **extra_specs):
    cls, specs, _ = CONFIGS[name]
    keras.utils.set_random_seed(0)
    meta = mneflow.load_meta(str(tfr_path), 'fc')
    # Training saves the model specs into the shared metadata file, so
    # start every model from empty specs
    meta.model_specs = {}
    meta.update(
        model_specs=dict(
            COMMON_SPECS, **specs, **extra_specs, model_id=model_id or name
        )
    )
    dataset = mneflow.Dataset(meta, train_batch=16)
    model = _quiet(cls, meta=meta, dataset=dataset, specs_prefix=False)
    _quiet(
        model.build,
        loss='MAE',
        metrics='RootMeanSquaredError',
        learn_rate=1e-2,
    )
    return model


def train_cv(model, n_epochs=2, eval_step=3, early_stopping=1):
    _quiet(
        model.train,
        n_epochs=n_epochs,
        eval_step=eval_step,
        early_stopping=early_stopping,
        mode='cv',
        collect_patterns=isinstance(model, SymmetricModel),
    )


def read_log(model):
    path = os.path.join(model.model_path, model.scope + '_log.csv')
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_list(text):
    return [
        float(x)
        for x in text.replace('np.float64(', '')
        .replace(')', '')
        .strip('[]')
        .split(',')
        if x.strip()
    ]


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
    rows = [
        r for r in read_log(model) if r['spec_model_id'] == f'log_test_{name}'
    ]
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
    assert os.path.exists(
        os.path.join(model.model_path, model.model_name + '.h5')
    )


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

    roi2, stability = _quiet(
        model.get_roi_labels, patterns=patterns, return_stability=True
    )
    assert set(roi2) == set(roi)
    for key in ('row', 'col', 'diag'):
        assert 0 <= stability[key]['jaccard'] <= 1
        assert 0 <= stability[key]['abs_r'] <= 1


def test_symmetric_patterns_without_trainable_pointwise(tfr_path):
    """With trainable_pointwise_kernel=False, one frequency weighting is
    shared by all components (pointwise weights (n_freqs, n_folds))."""
    model = make_model(tfr_path, 'symmetric_fixed_pointwise')
    train_cv(model)
    pointwise = model.meta.weights['pointwise']
    assert pointwise.shape == (N_FREQS, N_FOLDS)
    patterns = _quiet(model.extract_patterns)
    for key in ('row', 'col', 'diag'):
        assert patterns['spatial_patterns'][key].shape == (N_NODES, N_FOLDS)
    # The strongest frequency in each fold is that of the shared weights
    np.testing.assert_array_equal(
        patterns['freq_inds']['row'], np.argmax(pointwise, 0)
    )
    np.testing.assert_array_equal(
        patterns['freq_inds']['diag'], np.argmax(np.abs(pointwise), 0)
    )
    roi = _quiet(model.get_roi_labels, patterns=patterns)
    assert set(roi) == {'row', 'col', 'diag', 'row-col'}


def test_fold_stability_measures():
    rng = np.random.default_rng(0)
    pattern = rng.normal(size=(50, 1))
    same = np.repeat(pattern, 3, axis=1)
    stab = _fold_stability(same, threshold=1.0)
    assert stab['jaccard'] == pytest.approx(1)
    assert stab['abs_r'] == pytest.approx(1)

    unrelated = rng.normal(size=(5000, 3))
    stab = _fold_stability(unrelated, threshold=1.0)
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
    row = [
        r for r in read_log(model) if r['spec_model_id'] == f'learn_{name}'
    ][0]
    holdout_mae = np.mean(parse_list(row['test_losses']))

    _, y = make_data()
    test_fold = np.asarray(model.meta.data['test_fold'])
    train_y = np.delete(y, test_fold)
    baseline_mae = np.mean(np.abs(y[test_fold] - train_y.mean()))
    assert holdout_mae < 0.7 * baseline_mae


def test_fixed_pointwise_patterns_need_single_cross_filter(tfr_path):
    model = make_model(tfr_path, 'symmetric_fixed_pointwise', n_latent_cross=2)
    train_cv(model)
    with pytest.raises(NotImplementedError, match='n_latent_cross=1'):
        _quiet(model.extract_patterns)


def test_cv_training_stores_test_predictions(tfr_path):
    """With store_test_predictions, each fold model's predictions for the
    whole test set are kept, in test_fold order."""
    model = make_model(tfr_path, 'conv3d', model_id='pred_test')
    _quiet(
        model.train,
        n_epochs=2,
        eval_step=3,
        early_stopping=1,
        mode='cv',
        store_test_predictions=True,
    )
    _, y = make_data()
    test_fold = np.asarray(model.meta.data['test_fold'])
    assert len(model.cv_test_predictions) == N_FOLDS
    for fold in model.cv_test_predictions:
        np.testing.assert_allclose(
            np.ravel(fold['y_true']), y[test_fold], rtol=1e-5
        )
        assert np.ravel(fold['y_pred']).shape == test_fold.shape


def _produce(path, inputs, test_set='holdout'):
    np.random.seed(0)
    return _quiet(
        mneflow.produce_tfrecords,
        inputs,
        path=str(path),
        data_id='fc',
        fs=1,
        input_type='fconn',
        target_type='float',
        n_folds=N_FOLDS,
        overwrite=True,
        test_set=test_set,
    )


def test_loso_test_predictions_cover_each_subject(tmp_path):
    """With loso and subjects of different sizes, each fold's test
    predictions cover exactly the held-out subject's samples."""
    # Unequal sizes, not multiples of N_FOLDS; the first is larger than
    # the test batch (100)
    data = [make_data(seed=0), make_data(seed=1)]
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([d[1] for d in data])
    sizes = [157, 35]
    inputs = [(X[:157], y[:157]), (X[157:], y[157:])]
    _produce(tmp_path, inputs, test_set='loso')
    model = make_model(tmp_path, 'conv3d', model_id='loso_pred')
    _quiet(
        model.train,
        n_epochs=1,
        eval_step=2,
        early_stopping=1,
        mode='loso',
        store_test_predictions=True,
    )
    assert len(model.cv_test_predictions) == len(inputs)
    for (_, y_subj), size, fold in zip(
        inputs, sizes, model.cv_test_predictions
    ):
        assert np.ravel(fold['y_pred']).shape == (size,)
        np.testing.assert_allclose(
            np.sort(np.ravel(fold['y_true'])), np.sort(y_subj), rtol=1e-5
        )


def test_train_passes_shapley_order_to_collect_patterns(tfr_path):
    """Models whose collect_patterns takes shapley_order (e.g. LFCNN) get
    it from train; the connectivity models' collect_patterns does not
    take it (covered by the symmetric model tests)."""
    model = make_model(tfr_path, 'conv3d', model_id='shapley')
    calls = []

    def collect_patterns(fold=0, n_folds=1, shapley_order=0):
        calls.append(shapley_order)

    model.collect_patterns = collect_patterns
    _quiet(
        model.train,
        n_epochs=1,
        eval_step=2,
        early_stopping=1,
        mode='cv',
        collect_patterns=True,
        shapley_order=2,
    )
    assert calls == [2] * N_FOLDS


def test_log_with_old_columns_is_rewritten(tmp_path):
    """Appending to a log written with other columns (e.g. by an older
    mneflow without spec_model_id) keeps every row aligned."""
    _produce(tmp_path, make_data())
    model = make_model(tmp_path, 'conv3d', model_id='new_row')
    path = os.path.join(model.model_path, model.scope + '_log.csv')
    os.makedirs(model.model_path, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['data_id', 'model_id', 'loss'])
        writer.writeheader()
        writer.writerow({'data_id': 'old', 'model_id': 'm0', 'loss': 'MSE'})
    train_cv(model)
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = list(reader)
    assert header[:3] == ['data_id', 'model_id', 'loss']
    assert 'spec_model_id' in header
    assert all(len(line) == len(header) for line in lines)
    old, new = read_log(model)
    assert old['data_id'] == 'old' and old['loss'] == 'MSE'
    assert old['model_id'] == 'm0'
    assert old['spec_model_id'] == ''
    assert new['spec_model_id'] == 'new_row'
    assert new['model_id'] == model.model_name


@pytest.mark.parametrize('name', list(CONFIGS))
def test_no_channels_first_layers(tfr_path, name):
    """TensorFlow's CPU kernels support only channels_last convolutions
    (older versions fail on channels_first depthwise convolutions)."""
    model = make_model(tfr_path, name)
    formats = [
        getattr(layer, 'data_format', None) for layer in model.km.layers
    ]
    assert 'channels_first' not in formats


def test_trainable_pointwise_weights_each_component(tfr_path):
    """The trainable pointwise layer weights the last axis with a separate
    kernel for each component (axis 1), as a channels_first depthwise
    convolution would."""
    model = make_model(tfr_path, 'symmetric')
    rng = np.random.default_rng(0)
    inputs = rng.random((3,) + tuple(model.inputs.shape[1:]), np.float32)
    shape = tuple(model.squaresum_3d(inputs).shape)
    x = rng.normal(size=shape).astype(np.float32)
    kernel = rng.normal(size=model.pointwise.kernel.shape).astype(np.float32)
    bias = rng.normal(size=model.pointwise.bias.shape).astype(np.float32)
    model.pointwise.set_weights([kernel, bias])
    expected = np.einsum('bahw,wa->bah', x, kernel[0, :, :, 0])
    expected = np.maximum(expected + bias[None, :, None], 0)[..., None]
    got = keras.ops.convert_to_numpy(model._apply_pointwise(x))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
