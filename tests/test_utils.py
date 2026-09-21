import numpy as np
import pytest

from mneflow.utils import r2_score, _onehot, cosine_similarity


def test_r2_score_perfect_prediction():
    # Arrange: known inputs where we can compute the expected answer by hand
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    # Act: call the function under test
    result = r2_score(y_true, y_pred)

    # Assert: check the result is what we expect
    assert result == pytest.approx(1.0)


def test_r2_score_worse_than_mean():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # anti-correlated
    assert r2_score(y_true, y_pred) < 0


def test_onehot_basic():
    y = np.array([0, 1, 2, 1])
    encoded = _onehot(y, n_classes=3)
    assert encoded.shape == (4, 3)
    # row 0 should be class 0 -> [1, 0, 0]
    np.testing.assert_array_equal(encoded[0], [1, 0, 0])


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)
