"""
Tier C: tests for mneflow.MetaData's own persistence -- save() writing
a pickle to disk, the pickle round trip preserving every attribute,
update()'s merge-then-resave behavior, and change_dataset()'s
metadata-borrowing logic.

This is deliberately separate from tier B (BaseModel.save() +
MetaData.restore_model(), the model-weights save/load path -- see the
"Update" note in the Tier 1 section of this doc for why that one isn't
unit-testable with fakes): MetaData itself has no TensorFlow
dependency at all (meta.py only imports mne/numpy/scipy/matplotlib/
pickle), and save()/update()/change_dataset() are pure Python plus
filesystem I/O, so they're straightforward to test directly with
tmp_path and no real Dataset, model, or TFRecords involved.
"""
import os
import pickle

import numpy as np
import pytest

from mneflow.meta import MetaData


def _populated_meta(path, data_id='subj01'):
    """A MetaData instance with representative, non-empty values in
    every attribute -- including a numpy array in `weights`, since
    that's the attribute most likely to break under pickling if
    something non-picklable ever sneaks into it.
    """
    meta = MetaData()
    meta.data = {'path': str(path), 'data_id': data_id,
                 'input_type': 'trials', 'target_type': 'int',
                 'n_folds': 5}
    meta.preprocessing = {'scale': True, 'decimate': None}
    meta.model_specs = {'scope': 'lfcnn', 'n_latent': 32, 'dropout': 0.5}
    meta.train_params = {'optimizer': 'adam', 'learn_rate': 3e-4}
    meta.patterns = {'dmx': np.arange(12).reshape(3, 4).astype(np.float64)}
    meta.results = {'v_metric': 0.87, 'v_loss': 0.31}
    meta.weights = {'dmx': np.random.default_rng(0).normal(size=(4, 3))}
    return meta


# --- save() ---

def test_save_writes_pickle_file_named_by_data_id(tmp_path):
    meta = _populated_meta(tmp_path, data_id='subj01')
    meta.save(verbose=False)

    expected = tmp_path / 'subj01_meta.pkl'
    assert expected.exists()


def test_save_is_a_noop_without_path_or_data_id(tmp_path):
    """save() only ever prints a message and returns when 'path' or
    'data_id' is missing from meta.data -- it never raises, and it
    never writes a file. This pins that down so a future refactor
    that makes it raise instead is a deliberate, visible change.
    """
    meta = MetaData()
    meta.data = {'input_type': 'trials'}  # no 'path', no 'data_id'

    meta.save(verbose=False)  # must not raise

    assert list(tmp_path.iterdir()) == []


# --- pickle round trip ---

def test_pickle_round_trip_preserves_every_attribute(tmp_path):
    meta = _populated_meta(tmp_path, data_id='subj02')
    meta.save(verbose=False)

    with open(tmp_path / 'subj02_meta.pkl', 'rb') as f:
        restored = pickle.load(f)

    assert restored.data == meta.data
    assert restored.preprocessing == meta.preprocessing
    assert restored.model_specs == meta.model_specs
    assert restored.train_params == meta.train_params
    assert restored.results == meta.results

    # dict equality doesn't compare numpy arrays element-wise the way
    # you'd expect (`==` on arrays returns an array, not a bool), so
    # patterns/weights need their own explicit check.
    assert restored.patterns.keys() == meta.patterns.keys()
    np.testing.assert_array_equal(restored.patterns['dmx'], meta.patterns['dmx'])
    assert restored.weights.keys() == meta.weights.keys()
    np.testing.assert_array_equal(restored.weights['dmx'], meta.weights['dmx'])


# --- update() ---

def test_update_merges_into_existing_dicts_without_clobbering(tmp_path):
    """update() should MERGE the given dict into the attribute, not
    replace it -- keys already present and not mentioned in the call
    must survive.
    """
    meta = _populated_meta(tmp_path, data_id='subj03')

    meta.update(model_specs={'dropout': 0.25, 'n_latent': 64})

    # New/changed keys took effect...
    assert meta.model_specs['dropout'] == 0.25
    assert meta.model_specs['n_latent'] == 64
    # ...but 'scope', which wasn't mentioned, must still be there.
    assert meta.model_specs['scope'] == 'lfcnn'


def test_update_persists_the_merge_to_disk(tmp_path):
    """update() calls self.save() internally -- the merged state
    should be what a fresh pickle.load() sees, not just what's held
    in memory.
    """
    meta = _populated_meta(tmp_path, data_id='subj04')

    meta.update(results={'v_metric': 0.99})

    with open(tmp_path / 'subj04_meta.pkl', 'rb') as f:
        restored = pickle.load(f)
    assert restored.results['v_metric'] == 0.99
    # 'v_loss' wasn't touched by this update() call -- merge, not replace.
    assert restored.results['v_loss'] == 0.31


def test_update_ignores_non_dict_arguments(tmp_path):
    """Each argument is merged only `if isinstance(arg, dict)` -- passing
    None (the default) or any non-dict value for an attribute you don't
    want to touch is a deliberate no-op, not an error.
    """
    meta = _populated_meta(tmp_path, data_id='subj05')
    original_preprocessing = dict(meta.preprocessing)

    meta.update(preprocessing=None, train_params="not a dict")

    assert meta.preprocessing == original_preprocessing
    assert meta.train_params == {'optimizer': 'adam', 'learn_rate': 3e-4}


# --- change_dataset() ---

def test_change_dataset_copies_only_the_documented_keys(tmp_path):
    """change_dataset() loads another dataset's saved MetaData and
    copies over ONLY the dataset-identity keys it documents -- not
    the whole `.data` dict. Note this list includes 'path' and
    'data_id' themselves, so calling this also repoints `self` at the
    other dataset's save location, which is easy to miss when reading
    the method's one-line docstring summary.
    """
    other = MetaData()
    other.data = {
        'path': str(tmp_path), 'data_id': 'other_subj',
        'data_path': '/raw/other', 'test_set': 'holdout',
        'train_paths': ['a.tfrecord'], 'test_paths': ['b.tfrecord'],
        'folds': [[0, 1], [2, 3]], 'indices': [0, 1, 2, 3],
        'n_folds': 5, 'test_fold': 0, 'train_size': 100,
        'test_size': 20, 'val_size': 20,
        'n_seq': 1, 'n_t': 64, 'n_ch': 16, 'y_shape': (2,),
        # A key deliberately NOT in change_dataset()'s copy list --
        # must NOT leak into `self.data` below.
        'input_type': 'trials',
    }
    other.save(verbose=False)

    meta = MetaData()
    meta.data = {'path': str(tmp_path), 'data_id': 'self_subj',
                 'input_type': 'continuous'}

    meta.change_dataset('other_subj', meta_path=str(tmp_path))

    assert meta.data['data_id'] == 'other_subj'  # copied, repoints self
    assert meta.data['path'] == str(tmp_path)
    assert meta.data['n_t'] == 64
    assert meta.data['y_shape'] == (2,)
    # Not in the documented key list -- change_dataset() must leave it
    # alone rather than copying it over from `other`.
    assert meta.data['input_type'] == 'continuous'


def test_change_dataset_uses_self_path_when_meta_path_omitted(tmp_path):
    other = MetaData()
    other.data = {
        'path': str(tmp_path), 'data_id': 'other_subj2',
        'data_path': '/raw/other', 'test_set': None,
        'train_paths': [], 'test_paths': [], 'folds': [], 'indices': [],
        'n_folds': 5, 'test_fold': 0, 'train_size': 0, 'test_size': 0,
        'val_size': 0, 'n_seq': 1, 'n_t': 32, 'n_ch': 8, 'y_shape': (4,),
    }
    other.save(verbose=False)

    meta = MetaData()
    meta.data = {'path': str(tmp_path), 'data_id': 'self_subj2'}

    # meta_path omitted -> falls back to self.data['path'], which is
    # tmp_path (same directory `other` was just saved into).
    meta.change_dataset('other_subj2')

    assert meta.data['n_ch'] == 8


def test_change_dataset_finds_file_without_trailing_separator_in_meta_path(tmp_path):
    """Regression test for a real bug: change_dataset() used to build
    the metadata file's path with plain string concatenation
    (`meta_path + new_data_id + '_meta.pkl'`) instead of
    os.path.join(), so a meta_path without a trailing separator (the
    normal way to pass a directory, and what save() itself produces
    via os.path.join()) silently failed to find a file that was
    genuinely there -- os.path.exists() just returned False, and the
    method then crashed with UnboundLocalError on 'new_meta' a few
    lines later, instead of raising anything that points at the real
    cause. Now fixed to use os.path.join() like save() does.
    """
    other = MetaData()
    other.data = {
        'path': str(tmp_path), 'data_id': 'other_subj3',
        'data_path': '/raw/other', 'test_set': None,
        'train_paths': [], 'test_paths': [], 'folds': [], 'indices': [],
        'n_folds': 5, 'test_fold': 0, 'train_size': 0, 'test_size': 0,
        'val_size': 0, 'n_seq': 1, 'n_t': 32, 'n_ch': 8, 'y_shape': (4,),
    }
    other.save(verbose=False)

    meta = MetaData()
    meta.data = {'path': str(tmp_path), 'data_id': 'self_subj3'}

    # str(tmp_path) has NO trailing separator -- exactly the shape of
    # path that used to break this method.
    assert not str(tmp_path).endswith(os.sep)
    meta.change_dataset('other_subj3', meta_path=str(tmp_path))

    assert meta.data['n_ch'] == 8
