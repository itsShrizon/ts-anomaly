import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCE(nn.Module):
    """anomalies are rare. focal loss helps the tail.

    numerically stable form: uses log-sigmoid rather than sigmoid + log.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # bce with logits already stable
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        # p_t in log-space: avoids sigmoid on large |logits|
        logp = -bce  # = log p_t
        pt = logp.exp().clamp(max=1.0)
        w = torch.where(y > 0.5, self.alpha, 1 - self.alpha)
        return (w * (1 - pt).pow(self.gamma) * bce).mean()
