import numpy as np

from src.infer.smooth import median_smooth


def test_kills_single_spike():
    x = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    out = median_smooth(x, k=3)
    assert out[3] == 0


def test_k1_identity():
    x = np.random.randn(10)
    np.testing.assert_array_equal(median_smooth(x, k=1), x)
