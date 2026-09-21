import numpy as np
import pytest
import tensorflow as tf

from mneflow.layers import (
    DeMixing,
    FullyConnected,
    LFTConv,
    VARConv,
    TempPooling,
    SquareSymm,
    WeightedSum,
    WeightSum3d,
    SquareSum3d,
    LSTM,
)

# Every BaseLayer subclass's build() calls _set_constraints() then
# _set_regularizer(), and BOTH read straight out of `specs` with no
# fallback: `self.specs['unitnorm_scope']`, `self.specs['l1_scope']`,
# `self.specs['l2_scope']` (layers.py's BaseLayer, around line 90-120).
# Every model class in models.py/lfcnn.py (LFCNN, VARCNN, ...) honors
# this by setting all three via `meta.model_specs.setdefault(...)`
# before it ever constructs a layer -- so when testing layers directly,
# bypassing any model class, we have to supply the same minimum
# ourselves. Without it, the *first* layer built raises
# `KeyError: 'unitnorm_scope'` (checked before 'l1_scope'/'l2_scope',
# since _set_constraints runs first).
DEFAULT_LAYER_SPECS = {'l1_scope': [], 'l2_scope': [], 'unitnorm_scope': []}


@pytest.fixture
def eeg_batch():
    """Synthetic (batch, 1, n_times, n_channels) input, mneflow's NHWC convention."""
    rng = np.random.default_rng(seed=0)
    batch, n_times, n_channels = 4, 64, 16
    x = rng.normal(size=(batch, 1, n_times, n_channels)).astype(np.float32)
    return tf.constant(x)


@pytest.mark.parametrize("n_latent", [1, 8, 16])
def test_demixing_output_shape(eeg_batch, n_latent):
    layer = DeMixing(size=n_latent, specs=DEFAULT_LAYER_SPECS)
    out = layer(eeg_batch)
    batch, h, t, _ = eeg_batch.shape
    assert out.shape == (batch, h, t, n_latent)


@pytest.mark.parametrize("n_units", [1, 10, 32])
def test_fullyconnected_output_shape(eeg_batch, n_units):
    layer = FullyConnected(size=n_units, specs=DEFAULT_LAYER_SPECS)
    out = layer(eeg_batch)
    assert out.shape == (eeg_batch.shape[0], n_units)


@pytest.mark.parametrize("filter_length", [3, 7, 15])
def test_lftconv_preserves_shape(eeg_batch, filter_length):
    # depthwise + SAME padding + stride 1 -> shape unchanged
    layer = LFTConv(filter_length=filter_length, specs=DEFAULT_LAYER_SPECS)
    out = layer(eeg_batch)
    assert out.shape == eeg_batch.shape


@pytest.mark.parametrize("n_filters", [4, 8])
def test_varconv_output_shape(eeg_batch, n_filters):
    layer = VARConv(size=n_filters, filter_length=7, specs=DEFAULT_LAYER_SPECS)
    out = layer(eeg_batch)
    batch, h, t, _ = eeg_batch.shape
    assert out.shape == (batch, h, t, n_filters)


@pytest.mark.parametrize("stride,pool_type", [(2, "max"), (2, "avg"), (4, "max")])
def test_temppooling_output_shape(eeg_batch, stride, pool_type):
    # TempPooling.build() has no trainable weights and never calls
    # _set_constraints()/_set_regularizer() -- no specs needed.
    layer = TempPooling(stride=stride, pooling=stride, pool_type=pool_type)
    out = layer(eeg_batch)
    batch, h, t, c = eeg_batch.shape
    expected_t = int(np.ceil(t / stride))  # SAME padding
    assert out.shape == (batch, h, expected_t, c)


# --- SquareSymm: congruence transform on a square (batch, N, N) input ---

@pytest.fixture
def square_batch():
    """Synthetic square per-sample matrices, e.g. spatial covariance-like."""
    rng = np.random.default_rng(seed=1)
    batch, n = 4, 12
    x = rng.normal(size=(batch, n, n)).astype(np.float32)
    return tf.constant(x)


@pytest.mark.parametrize("size", [1, 5, 12, 20])
def test_squaresymm_output_shape(square_batch, size):
    layer = SquareSymm(size=size, specs=DEFAULT_LAYER_SPECS)
    out = layer(square_batch)
    batch = square_batch.shape[0]
    assert out.shape == (batch, size, size)


def test_squaresymm_raises_on_non_square_input():
    # input_shape[1] != input_shape[-1] -> the second tensordot in call()
    # contracts two axes of mismatched size and must fail.
    rng = np.random.default_rng(seed=2)
    x = tf.constant(rng.normal(size=(4, 12, 7)).astype(np.float32))
    layer = SquareSymm(size=8, specs=DEFAULT_LAYER_SPECS)
    with pytest.raises(Exception):
        layer(x)


# --- WeightedSum: collapse one axis into `size` components ---

@pytest.mark.parametrize("size", [1, 8, 16])
def test_weightedsum_output_shape_2d_input(size):
    rng = np.random.default_rng(seed=3)
    batch, n_rows = 4, 20
    x = tf.constant(rng.normal(size=(batch, n_rows)).astype(np.float32))
    layer = WeightedSum(size=size, axis=1, specs=DEFAULT_LAYER_SPECS)
    out = layer(x)
    assert out.shape == (batch, size)


# --- WeightSum3d: per-channel weighted sum, one weight matrix per channel ---

@pytest.mark.parametrize("size", [1, 4, 10])
def test_weightsum3d_output_shape(size):
    rng = np.random.default_rng(seed=4)
    batch, n_rows, n_channels = 4, 20, 6
    x = tf.constant(rng.normal(size=(batch, n_rows, n_channels)).astype(np.float32))
    layer = WeightSum3d(size=size, axis=1, specs=DEFAULT_LAYER_SPECS)
    out = layer(x)
    assert out.shape == (batch, size, n_channels)


# --- SquareSum3d: per-channel congruence transform; requires size == n_rows ---

def test_squaresum3d_output_shape_when_size_matches_rows():
    rng = np.random.default_rng(seed=5)
    batch, n_rows, n_channels = 4, 10, 6
    x = tf.constant(rng.normal(size=(batch, n_rows, n_channels)).astype(np.float32))
    layer = SquareSum3d(size=n_rows, axis=1, specs=DEFAULT_LAYER_SPECS)  # size must equal n_rows
    out = layer(x)
    assert out.shape == (batch, n_rows, n_channels)


def test_squaresum3d_raises_when_size_does_not_match_rows():
    rng = np.random.default_rng(seed=6)
    batch, n_rows, n_channels = 4, 10, 6
    x = tf.constant(rng.normal(size=(batch, n_rows, n_channels)).astype(np.float32))
    layer = SquareSum3d(size=n_rows + 3, axis=1, specs=DEFAULT_LAYER_SPECS)  # mismatched on purpose
    with pytest.raises(Exception):
        layer(x)


# --- LSTM: thin wrapper around tf.keras.layers.LSTM (no `specs` param at all) ---

@pytest.mark.parametrize("size,return_sequences", [(8, True), (8, False), (16, True)])
def test_lstm_output_shape(size, return_sequences):
    rng = np.random.default_rng(seed=7)
    batch, n_times, n_features = 4, 30, 5
    x = tf.constant(rng.normal(size=(batch, n_times, n_features)).astype(np.float32))
    layer = LSTM(size=size, return_sequences=return_sequences)
    out = layer(x)
    if return_sequences:
        assert out.shape == (batch, n_times, size)
    else:
        assert out.shape == (batch, size)
