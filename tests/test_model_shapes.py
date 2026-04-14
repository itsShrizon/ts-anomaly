import torch

from src.models.hybrid import HybridAnomaly


def test_forward_shape():
    m = HybridAnomaly(in_dim=8, hidden=16, rnn_layers=1, heads=2, attn_layers=1)
    x = torch.randn(4, 32, 8)
    y = m(x)
    assert y.shape == (4,)


def test_batch_one():
    m = HybridAnomaly(in_dim=4, hidden=8, rnn_layers=1, heads=2, attn_layers=1)
    y = m(torch.randn(1, 16, 4))
    assert y.shape == (1,)
