"""Smoke tests for the custom loss functions in mneflow.losses."""

import numpy as np
import pytest
import tensorflow as tf

from mneflow import losses

# Losses combining a cosine similarity along the time axis with MSE/MAE,
# for targets of shape (batch, 1, times, channels)
SIGNAL_LOSSES = [
    'Cos2MSE',
    'Cos2_3',
    'Cos2MAE',
    'CosMSE',
    'Cos23MSE',
    'Cos3MAE',
    'MSAE',
    'dMSAE',
]


@pytest.mark.parametrize('name', SIGNAL_LOSSES)
def test_signal_losses_return_finite_values(name):
    rng = np.random.default_rng(0)
    shape = (8, 1, 30, 4)
    y_true = tf.constant(rng.normal(size=shape).astype(np.float32))
    y_pred = tf.constant(rng.normal(size=shape).astype(np.float32))
    loss = getattr(losses, name)(y_true, y_pred)
    assert np.all(np.isfinite(loss))


@pytest.mark.parametrize('name', ['MSAE', 'dMSAE'])
def test_msae_losses_are_zero_for_perfect_predictions(name):
    y = tf.constant(
        np.random.default_rng(0).normal(size=(8, 1, 30, 4)).astype(np.float32)
    )
    np.testing.assert_allclose(getattr(losses, name)(y, y), 0, atol=1e-6)
