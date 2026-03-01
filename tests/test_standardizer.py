import numpy as np
from src.data.scaling import Standardizer


def test_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5, scale=3, size=(1000, 4)).astype(np.float32)
    sc = Standardizer.fit(x)
    z = sc.transform(x)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-4)


def test_constant_channel_safe():
    x = np.ones((100, 3), dtype=np.float32)
    sc = Standardizer.fit(x)
    # should not explode on zero std
    z = sc.transform(x)
    assert np.isfinite(z).all()
