"""
Round-trip tests for mneflow's custom Keras layers' get_config()/
from_config() -- the path tf.keras.models.load_model() uses to
reconstruct a saved model's architecture from scratch.

This is a DIFFERENT round trip from the one
mneflow.MetaData.restore_model() actually uses in practice (which
rebuilds the architecture from Python code and only load_weights()s
the saved weights -- see test_models.py). get_config()/from_config()
is exercised only if something calls tf.keras.models.load_model()
directly on a saved .h5, but every custom layer is decorated
`@saving.register_keras_serializable`, which promises that path works.

Two bugs were found and fixed here:

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
   build(). Tracked below as test_varconv_from_config_still_loses_size
   (LFTConv is exempt from an equivalent assertion since a wrong
   'size' there has no observable effect).

Two classes were deliberately left out of the fix in layers.py and
have no tests here:

- LFTConvTranspose's from_config() has the same `cls(nonlin, **config)`
  shape, but its __init__'s first positional parameter is
  `target_shape`, not `scope` -- a different, unrelated bug in a class
  whose scope/nonlin/specs constructor arguments are commented out
  entirely. Left alone.
- LSTM's from_config() also has the same shape, but a deeper bug on
  top of it: get_config() merges the base tf.keras.layers.LSTM config
  (which already has 'units'/'activation') with the custom
  'size'/'nonlin' keys, so the same dict carries both names for the
  same value. Passing that combined dict back into `cls(**config)` --
  whether nonlin lands positionally or by keyword -- always raises
  `TypeError: got multiple values for keyword argument 'units'` once
  the leftover 'units'/'activation' keys reach LSTM.__init__'s
  **args and collide with its own explicit `units=size,
  activation=nonlin` passed to super(). This predates the fix here
  and isn't addressed by it.
"""
import numpy as np
import pytest
import tensorflow as tf

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


# --- Layers whose get_config() captures every constructor argument ---
# (DeMixing, FullyConnected, SquareSymm, WeightedSum, WeightSum3d,
# SquareSum3d) -- so a full round trip through get_config()/from_config()
# should now reproduce the layer's scope and nonlin exactly.

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
    assert config['nonlin'] is kwargs['nonlin']

    restored = cls.from_config(config)

    # from_config() now passes scope/nonlin back in by keyword
    # (`cls(scope=scope, nonlin=nonlin, **config)`), so both survive
    # the round trip intact.
    assert restored.scope == kwargs['scope']
    assert restored.nonlin is kwargs['nonlin']
    assert restored.size == kwargs['size']


# --- LFTConv / VARConv: scope/nonlin now round-trip correctly (bug 1,
# fixed), but get_config() still never includes 'size' (bug 2, not
# fixed) -- from_config() still reconstructs with the class default.

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

    # Bug 1 (fixed): scope/nonlin now survive the round trip.
    assert restored.scope == kwargs['scope']
    assert restored.nonlin is kwargs['nonlin']
    assert restored.filter_length == kwargs['filter_length']
    assert restored.padding == kwargs['padding']


def test_varconv_from_config_still_loses_size():
    """Pins down bug 2 specifically for VARConv, where a lost `size`
    is not cosmetic: it sets the number of output convolution filters
    used directly in build(). A model reloaded via
    tf.keras.models.load_model() today would silently get 32 output
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
