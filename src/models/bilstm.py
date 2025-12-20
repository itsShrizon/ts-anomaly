import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers,
                           batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out
