# -*- coding: utf-8 -*-
"""
mneflow._compat
================
Single place to shim APIs whose location, name, or signature changes
across the numpy / scipy / mne / tensorflow-keras versions mneflow
supports (see README "Supported / tested versions" for the tested
range on each Python version).

Why this file exists
---------------------
``tf.keras.ops`` is the unified Keras-3 tensor-ops namespace. It only
exists when TensorFlow is running with Keras 3 as its backend, which
first shipped in TensorFlow 2.16. mneflow's declared floor
(``tensorflow>=2.12.0``, still Keras 2) does NOT have ``tf.keras.ops``,
so calling it directly raises ``AttributeError`` on tensorflow
2.12-2.15. A handful of call sites in ``fc_models.py``, ``lfcnn.py``
and ``models.py`` used ``tf.keras.ops.*`` directly; they now import
the wrappers below instead.

The point of keeping this in one module: the next time an upstream
package relocates or renames something mneflow depends on, the fix is
one function in this file, not a re-audit of every call site across
the package.

Add new shims here following the same pattern: try the current/newer
API first, fall back to the older equivalent, and keep the wrapper's
call signature identical to the newer API so callers don't need to
know which branch ran.
"""
import tensorflow as tf

# True when this TensorFlow build exposes the Keras-3 ``tf.keras.ops``
# namespace (TensorFlow >= 2.16, Keras 3 backend). False on Keras-2-only
# TensorFlow (2.12 - 2.15), where the equivalent raw TF ops are used
# instead.
KERAS_OPS_AVAILABLE = hasattr(tf.keras, "ops")


def ops_mean(x, axis=None, keepdims=False):
    """``keras.ops.mean`` with a Keras-2-compatible (``tf.reduce_mean``)
    fallback."""
    if KERAS_OPS_AVAILABLE:
        return tf.keras.ops.mean(x, axis=axis, keepdims=keepdims)
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def ops_split(x, indices_or_sections, axis=0):
    """``keras.ops.split`` with a Keras-2-compatible (``tf.split``)
    fallback.

    Only the integer form of ``indices_or_sections`` (split into that
    many equal-size chunks along ``axis``) is supported, which is all
    mneflow currently needs; ``tf.split``'s ``num_or_size_splits``
    accepts the same integer semantics.
    """
    if KERAS_OPS_AVAILABLE:
        return tf.keras.ops.split(x, indices_or_sections=indices_or_sections,
                                   axis=axis)
    return tf.split(x, num_or_size_splits=indices_or_sections, axis=axis)


def ops_expand_dims(x, axis):
    """``keras.ops.expand_dims`` with a Keras-2-compatible
    (``tf.expand_dims``) fallback."""
    if KERAS_OPS_AVAILABLE:
        return tf.keras.ops.expand_dims(x, axis)
    return tf.expand_dims(x, axis)


def ops_concatenate(tensors, axis=-1):
    """``keras.ops.concatenate`` with a Keras-2-compatible
    (``tf.concat``) fallback."""
    if KERAS_OPS_AVAILABLE:
        return tf.keras.ops.concatenate(tensors, axis=axis)
    return tf.concat(tensors, axis=axis)


def ops_transpose(x, axes=None):
    """``keras.ops.transpose`` with a Keras-2-compatible
    (``tf.transpose``) fallback."""
    if KERAS_OPS_AVAILABLE:
        return tf.keras.ops.transpose(x, axes=axes)
    return tf.transpose(x, perm=axes)
