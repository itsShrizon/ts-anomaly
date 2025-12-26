import torch
import torch.nn as nn


class FocalBCE(nn.Module):
    """anomalies are rare. focal loss helps the tail."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(y > 0.5, p, 1 - p)
        w = self.alpha * (1 - pt).pow(self.gamma)
        return (w * bce).mean()
