import torch
import torch.nn as nn

from .pos import SinusoidalPE


class TransformerStack(nn.Module):
    def __init__(self, d: int, heads: int = 4, layers: int = 2, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pe = SinusoidalPE(d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ff, dropout=dropout,
                                         batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(self.pe(x))
