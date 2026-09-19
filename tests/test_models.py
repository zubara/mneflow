"""
Tier-1 tests: architecture-wiring / shape tests for mneflow's models.

BaseModel.__init__ (models.py) builds the real Keras graph immediately
(`self.y_pred = self.build_graph()`), but only reads a handful of
attributes off the `meta`/`dataset` objects it's given to get there --
it never needs a real mneflow.Dataset or TFRecords on disk unless you
go on to call `.build()`/`.train()`, which these tests never do.
A lightweight duck-typed stand-in for each is enough to exercise the
real layer wiring and check the resulting output shape.
"""
import numpy as np
import pytest
import tensorflow as tf

from mneflow.models import BaseModel
from mneflow.lfcnn import LFCNN


class FakeDataset:
    """Duck-typed stand-in for mneflow.Dataset.

    BaseModel.__init__ only reads `.h_params` (a handful of keys) and
    `.y_shape` off the dataset it's given.
    """

    def __init__(self, n_seq, n_t, n_ch, y_shape, channel_subset=None,
                 target_type='int'):
        self.h_params = {
            'n_seq': n_seq,
            'n_t': n_t,
            'n_ch': n_ch,
            'channel_subset': channel_subset,
            'target_type': target_type,
        }
        self.y_shape = y_shape


class FakeMeta:
    """Duck-typed stand-in for mneflow.MetaData.

    BaseModel.__init__ only reads/writes `.model_specs` (a plain dict)
    and `.data['path']` / `.data['data_id']`.
    """

    def __init__(self, path, data_id='faketest', model_specs=None):
        self.data = {'path': str(path), 'data_id': data_id}
        self.model_specs = dict(model_specs or {})


@pytest.fixture
def meta_and_dataset(tmp_path):
    """Factory fixture: build a (meta, dataset) pair for a given shape.

    `tmp_path` is pytest's built-in per-test temp directory --
    BaseModel.__init__ creates a `models/` subfolder under
    `meta.data['path']`, so this keeps that off your real repo.
    """
    def _make(n_seq=1, n_t=64, n_ch=16, n_classes=2, model_specs=None):
        meta = FakeMeta(tmp_path, model_specs=model_specs or {})
        dataset = FakeDataset(n_seq=n_seq, n_t=n_t, n_ch=n_ch,
                              y_shape=(n_classes,))
        return meta, dataset
    return _make


# --- BaseModel: the default architecture (Flatten -> FullyConnected) ---
#
# BaseModel has no subclass-level `setdefault`s for its specs, unlike
# LFCNN etc below -- its default build_graph() still needs these three
# keys for FullyConnected's regularizer/constraint lookup to not KeyError.
MIN_BASEMODEL_SPECS = {'l1_scope': [], 'l2_scope': [], 'unitnorm_scope': []}


@pytest.mark.parametrize("n_seq,n_t,n_ch,n_classes", [
    (1, 64, 16, 2),
    (1, 100, 32, 4),
])
def test_basemodel_output_shape(meta_and_dataset, n_seq, n_t, n_ch, n_classes):
    meta, dataset = meta_and_dataset(n_seq=n_seq, n_t=n_t, n_ch=n_ch,
                                      n_classes=n_classes,
                                      model_specs=MIN_BASEMODEL_SPECS)
    model = BaseModel(meta, dataset=dataset)
    assert model.y_pred.shape == (None, n_classes)  # batch dim is symbolic
    assert model.input_shape == (n_seq, n_t, n_ch)


# --- LFCNN: DeMixing -> LFTConv -> TempPooling -> Dropout -> FullyConnected ---
#
# LFCNN.__init__ (lfcnn.py) fills in sensible defaults for most specs
# via `meta.model_specs.setdefault(...)`, EXCEPT 'dropout' -- build_graph()
# reads self.specs['dropout'] unconditionally but no setdefault ever
# provides it. See test_lfcnn_missing_dropout_spec_raises_keyerror below.

@pytest.mark.parametrize("n_latent,filter_length", [(8, 7), (16, 5), (4, 15)])
def test_lfcnn_output_shape(meta_and_dataset, n_latent, filter_length):
    meta, dataset = meta_and_dataset(
        n_t=64, n_ch=16, n_classes=3,
        model_specs={'dropout': 0.5, 'n_latent': n_latent,
                     'filter_length': filter_length})
    model = LFCNN(meta, dataset=dataset)
    assert model.y_pred.shape == (None, 3)


@pytest.mark.parametrize("pool_type,stride", [("max", 2), ("avg", 4)])
def test_lfcnn_output_shape_across_pooling_configs(meta_and_dataset, pool_type, stride):
    meta, dataset = meta_and_dataset(
        n_t=100, n_ch=8, n_classes=5,
        model_specs={'dropout': 0.5, 'pool_type': pool_type,
                     'stride': stride, 'pooling': stride})
    model = LFCNN(meta, dataset=dataset)
    assert model.y_pred.shape == (None, 5)


def test_lfcnn_missing_dropout_spec_raises_keyerror(meta_and_dataset):
    """Documents a real gap: LFCNN's `setdefault` calls never set a
    default for 'dropout', but build_graph() reads
    self.specs['dropout'] unconditionally. Constructing an LFCNN today
    without passing 'dropout' in specs raises KeyError. This test pins
    that down, so fixing it (adding
    `meta.model_specs.setdefault('dropout', ...)` in LFCNN.__init__)
    is a deliberate, visible change to this test -- not a silent one.
    """
    meta, dataset = meta_and_dataset(n_t=64, n_ch=16, n_classes=2)
    with pytest.raises(KeyError):
        LFCNN(meta, dataset=dataset)


# --- LFCNN initialization: does `setdefault` actually behave like `setdefault`? ---
#
# Two directions worth checking independently: omitted keys should come
# out to the documented default, and keys the caller DID supply should
# come out unchanged -- `setdefault` is a no-op when the key already
# exists, but that's exactly the kind of thing a refactor could get
# backwards without a test catching it.

def test_lfcnn_fills_in_documented_defaults_when_specs_partial(meta_and_dataset):
    """Only 'dropout' is supplied (required to dodge the KeyError bug
    above) -- every other key should come from LFCNN's own documented
    defaults (lfcnn.py's `setdefault` calls in __init__).
    """
    meta, dataset = meta_and_dataset(n_t=64, n_ch=16, n_classes=2,
                                      model_specs={'dropout': 0.5})
    model = LFCNN(meta, dataset=dataset)

    assert model.specs['filter_length'] == 7
    assert model.specs['n_latent'] == 32
    assert model.specs['pooling'] == 2
    assert model.specs['stride'] == 2
    assert model.specs['padding'] == 'SAME'
    assert model.specs['pool_type'] == 'max'
    assert model.specs['nonlin'] is tf.nn.relu
    assert model.specs['l1_lambda'] == pytest.approx(3e-4)
    assert model.specs['l2_lambda'] == pytest.approx(0.)
    assert model.specs['l1_scope'] == ['fc', 'dmx', 'tconv']
    assert model.specs['l2_scope'] == []
    assert model.specs['unitnorm_scope'] == []
    assert model.specs['scope'] == 'lfcnn'


def test_lfcnn_preserves_explicitly_given_specs(meta_and_dataset):
    """When every key IS supplied, `setdefault` must leave every one of
    them exactly as given -- none should be silently replaced by
    LFCNN's defaults.
    """
    custom_specs = {
        'dropout': 0.25,
        'filter_length': 11,
        'n_latent': 64,
        'pooling': 3,
        'stride': 3,
        'padding': 'VALID',
        'pool_type': 'avg',
        'nonlin': tf.nn.elu,
        'l1_lambda': 1e-2,
        'l2_lambda': 1e-3,
        'l1_scope': ['fc'],
        'l2_scope': ['dmx'],
        'unitnorm_scope': ['tconv'],
    }
    meta, dataset = meta_and_dataset(n_t=64, n_ch=16, n_classes=2,
                                      model_specs=dict(custom_specs))
    model = LFCNN(meta, dataset=dataset)

    for key, value in custom_specs.items():
        assert model.specs[key] == value, f"{key!r} was overwritten by a default"
