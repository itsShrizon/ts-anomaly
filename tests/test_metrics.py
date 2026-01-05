import numpy as np
from src.eval.metrics import auroc, best_f1


def test_auroc_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, s) == 1.0


def test_best_f1_threshold_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    f1, thr = best_f1(y, s)
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= thr <= 1.0
