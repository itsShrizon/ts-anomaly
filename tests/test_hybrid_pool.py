import torch

from src.models.hybrid import HybridAnomaly


def test_pool_last_vs_mean_differ():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 4)
    m_last = HybridAnomaly(4, 8, 1, 2, 1, pool="last")
    m_mean = HybridAnomaly(4, 8, 1, 2, 1, pool="mean")
    # copy weights so pooling is the only difference
    m_mean.load_state_dict(m_last.state_dict())
    assert not torch.allclose(m_last(x), m_mean(x))


def test_pool_invalid():
    try:
        HybridAnomaly(4, 8, pool="nope")
    except ValueError:
        return
    raise AssertionError("should have raised")
