"""BiLSTM -> Transformer hybrid. rnn captures local temporal, attn does long-range."""
import torch
import torch.nn as nn

from .bilstm import BiLSTMEncoder
from .transformer import TransformerStack


class HybridAnomaly(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, rnn_layers: int = 2,
                 heads: int = 4, attn_layers: int = 2, dropout: float = 0.1,
                 pool: str = "last"):
        super().__init__()
        if pool not in {"last", "mean"}:
            raise ValueError(f"pool must be last|mean, got {pool}")
        self.pool = pool
        self.rnn = BiLSTMEncoder(in_dim, hidden, rnn_layers, dropout)
        self.attn = TransformerStack(self.rnn.out_dim, heads, attn_layers, ff=hidden * 4, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.rnn.out_dim),
            nn.Linear(self.rnn.out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.rnn(x))
        z = h[:, -1] if self.pool == "last" else h.mean(dim=1)
        return self.head(z).squeeze(-1)
