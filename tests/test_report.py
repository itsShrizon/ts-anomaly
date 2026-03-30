import numpy as np
from src.eval.report import report


def test_report_keys():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = rng.random(400)
    r = report(y, s)
    assert set(r.keys()) == {"auroc", "best_f1", "best_threshold", "f1_at_chosen", "threshold_used"}
    assert 0 <= r["auroc"] <= 1
