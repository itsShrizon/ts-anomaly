import torch

from src.models.loss import FocalBCE


def test_focal_finite_at_extreme_logits():
    loss = FocalBCE()
    logits = torch.tensor([-50.0, 50.0, 0.0])
    y = torch.tensor([0.0, 1.0, 1.0])
    out = loss(logits, y)
    assert torch.isfinite(out)


def test_focal_reduces_weight_on_easy():
    loss = FocalBCE(alpha=0.5, gamma=2.0)
    # easy positive (large positive logit, label=1) should dominate less than a hard one
    easy = loss(torch.tensor([10.0]), torch.tensor([1.0]))
    hard = loss(torch.tensor([-1.0]), torch.tensor([1.0]))
    assert hard > easy
