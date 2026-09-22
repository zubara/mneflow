"""
Round-trip tests for mneflow's custom Keras layers' get_config()/
from_config() -- the path keras.models.load_model() uses to
reconstruct a saved model's architecture from scratch.

This is a DIFFERENT round trip from the one
mneflow.MetaData.restore_model() actually uses in practice (which
rebuilds the architecture from Python code and only load_weights()s
the saved weights -- see test_models.py). get_config()/from_config()
is exercised only if something calls keras.models.load_model()
directly on a saved .h5, but every custom layer is decorated
`@saving.register_keras_serializable`, which promises that path works.

Three bugs were found and fixed here:

1. (FIXED) Every custom layer's from_config() did
   `return cls(nonlin, **config)`, passing the deserialized activation
   function as the layer's FIRST POSITIONAL argument -- which is
   `scope` in every one of these __init__ signatures, not `nonlin`.
   The real `scope` string was popped out of `config` and discarded,
   and the real `nonlin` was never passed at all (it was consumed by
   the `scope` parameter instead), so the reconstructed layer silently
   fell back to its class's default nonlinearity. Fixed by passing
   both back in by keyword: `cls(scope=scope, nonlin=nonlin, **config)`,
   in DeMixing, FullyConnected, SquareSymm, LFTConv, VARConv,
   WeightedSum, WeightSum3d, and SquareSum3d. (WeightedSum and
   WeightSum3d had an extra variant of the same bug -- they assigned
   `cls.scope = config.pop("scope")`, a class attribute rather than a
   local variable, which is also fixed.)

2. (NOT FIXED) LFTConv and VARConv have a second, independent bug:
   their get_config() never includes 'size' at all, so from_config()
   still reconstructs the layer with size=32 (the class default)
   rather than whatever size it was actually built with. For LFTConv
   this is currently harmless (its docstring notes 'size' is unused by
   the layer's own weights); for VARConv it is not -- 'size' sets the
   number of output convolution filters and is used directly in
   build(). Tracked below as test_varconv_from_config_still_loses_size.

3. (FIXED) get_config() stored `'nonlin': self.nonlin` -- the raw,
   live Python function object -- directly in the config dict, and
   from_config() unconditionally called
   `saving.deserialize_keras_object(nonlin_config)` on it, which
   expects a *serialized* representation (a string or a config dict),
   not a live function. Calling get_config() then from_config()
   directly (as these tests do, and as keras.models.load_model()
   effectively does once the saved JSON has been parsed back into
   Python) raised `TypeError: Could not parse config: <function relu
   at 0x...>` for every layer, regardless of bug 1. This was masked
   during development because the sandbox this fix was written in has
   no TensorFlow installed, so nothing exercised it end to end until
   run for real. Fixed by serializing/deserializing `nonlin` properly,
   the way Keras activations are meant to be round-tripped:
   `'nonlin': keras.activations.serialize(self.nonlin)` in
   get_config(), and
   `nonlin = keras.activations.deserialize(nonlin_config)` in
   from_config(). One consequence: `keras.activations.deserialize`
   returns Keras's OWN function object for a built-in name (e.g.
   `keras.activations.relu`), not the exact object that was passed
   in (`tf.nn.relu`) -- they compute the same thing but are not the
   same Python object, so tests below compare by serialized name
   rather than by identity/`is`.

Two classes were deliberately left out of the fix in layers.py and
have no tests here:

- LFTConvTranspose's from_config() has the same `cls(nonlin, **config)`
  shape, but its __init__'s first positional parameter is
  `target_shape`, not `scope` -- a different, unrelated bug in a class
  whose scope/nonlin/specs constructor arguments are commented out
  entirely. Left alone.
- LSTM's from_config() also has the same shape (and got the same bug-3
  nonlin serialization fix, applied for consistency), but a deeper bug
  on top of it: get_config() merges the base keras.layers.LSTM
  config (which already has 'units'/'activation') with the custom
  'size'/'nonlin' keys, so the same dict carries both names for the
  same value. Passing that combined dict back into `cls(**config)` --
  regardless of bugs 1 or 3 -- always raises `TypeError: got multiple
  values for keyword argument 'units'` once the leftover
  'units'/'activation' keys reach LSTM.__init__'s **args and collide
  with its own explicit `units=size, activation=nonlin` passed to
  super(). This predates the fixes here and isn't addressed by them.
"""
import numpy as np
import pytest
import tensorflow as tf
import keras

from mneflow.layers import (
    DeMixing,
    FullyConnected,
    LFTConv,
    VARConv,
    SquareSymm,
    WeightedSum,
    WeightSum3d,
    SquareSum3d,
)

DEFAULT_LAYER_SPECS = {'l1_scope': [], 'l2_scope': [], 'unitnorm_scope': []}


def _built(layer, input_shape):
    """Force build() to run, the way a real model would, so
    get_config() reports values from an actually-built layer rather
    than a freshly-constructed one."""
    layer(tf.zeros(input_shape, dtype=tf.float32))
    return layer


def _same_activation(a, b):
    """True if two activation callables are the same Keras activation,
    even if they're different Python objects (e.g. tf.nn.relu vs.
    keras.activations.relu) -- compared by their canonical
    serialized name rather than by identity."""
    return keras.activations.serialize(a) == keras.activations.serialize(b)


# --- Layers whose get_config() captures every constructor argument ---
# (DeMixing, FullyConnected, SquareSymm, WeightedSum, WeightSum3d,
# SquareSum3d) -- so a full round trip through get_config()/from_config()
# should now reproduce the layer's scope and nonlin exactly (nonlin
# compared by activation identity, not object identity -- see bug 3).

@pytest.mark.parametrize("cls,kwargs,input_shape", [
    (DeMixing, dict(scope='dmx', size=8, nonlin=tf.nn.relu, axis=-1), (2, 1, 16, 6)),
    (FullyConnected, dict(scope='fc', size=5, nonlin=tf.nn.relu), (2, 16, 6)),
    (SquareSymm, dict(scope='ssym', size=4, nonlin=tf.nn.relu), (2, 6, 6)),
    (WeightedSum, dict(scope='wsum', size=3, nonlin=tf.nn.relu, axis=1), (2, 6)),
    (WeightSum3d, dict(scope='wsum3d', size=3, nonlin=tf.nn.relu, axis=1), (2, 6, 4)),
    (SquareSum3d, dict(scope='ssum3d', size=6, nonlin=tf.nn.relu, axis=1), (2, 6, 4)),
])
def test_from_config_round_trips_scope_and_nonlin(cls, kwargs, input_shape):
    layer = _built(cls(specs=DEFAULT_LAYER_SPECS, **kwargs), input_shape)
    config = layer.get_config()

    assert config['scope'] == kwargs['scope']
    # nonlin is now stored serialized (e.g. the string 'relu'), not as
    # the raw function object -- see bug 3.
    assert config['nonlin'] == keras.activations.serialize(kwargs['nonlin'])

    restored = cls.from_config(config)

    # from_config() now passes scope/nonlin back in by keyword
    # (`cls(scope=scope, nonlin=nonlin, **config)`), so both survive
    # the round trip.
    assert restored.scope == kwargs['scope']
    assert _same_activation(restored.nonlin, kwargs['nonlin'])
    assert restored.size == kwargs['size']


# --- LFTConv / VARConv: scope/nonlin now round-trip correctly (bugs 1
# and 3, fixed), but get_config() still never includes 'size' (bug 2,
# not fixed) -- from_config() still reconstructs with the class default.

@pytest.mark.parametrize("cls", [LFTConv, VARConv])
def test_lftconv_varconv_from_config_round_trips_scope_and_nonlin(cls):
    kwargs = dict(scope='tconv', size=8, nonlin=tf.nn.elu,
                  filter_length=5, padding='SAME')
    layer = _built(cls(specs=DEFAULT_LAYER_SPECS, **kwargs), (2, 1, 16, 6))
    config = layer.get_config()

    # Bug 2 (still present): 'size' never makes it into get_config().
    assert 'size' not in config
    # filter_length/padding round-trip correctly through get_config.
    assert config['filter_length'] == kwargs['filter_length']
    assert config['padding'] == kwargs['padding']

    restored = cls.from_config(config)

    # Bugs 1 and 3 (fixed): scope/nonlin now survive the round trip.
    assert restored.scope == kwargs['scope']
    assert _same_activation(restored.nonlin, kwargs['nonlin'])
    assert restored.filter_length == kwargs['filter_length']
    assert restored.padding == kwargs['padding']


def test_varconv_from_config_still_loses_size():
    """Pins down bug 2 specifically for VARConv, where a lost `size`
    is not cosmetic: it sets the number of output convolution filters
    used directly in build(). A model reloaded via
    keras.models.load_model() today would silently get 32 output
    filters regardless of how the original was built. LFTConv has the
    identical get_config() gap but no equivalent assertion here, since
    its own docstring notes 'size' is unused by that layer's weights.
    """
    layer = _built(
        VARConv(scope='tconv', size=8, nonlin=tf.nn.relu, filter_length=5,
                 specs=DEFAULT_LAYER_SPECS),
        (2, 1, 16, 6))
    config = layer.get_config()
    restored = VARConv.from_config(config)

    assert restored.size == 32  # class default -- NOT the original 8
