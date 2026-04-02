import numpy as np
import torch


@torch.no_grad()
def score(model, x: np.ndarray, device: str = "cpu", bs: int = 256) -> np.ndarray:
    """score a pre-windowed array using the torch model. inference Docker uses OrtScorer instead."""
    model.eval().to(device)
    xt = torch.from_numpy(x.astype(np.float32))
    chunks: list[np.ndarray] = []
    for i in range(0, len(xt), bs):
        logits = model(xt[i:i + bs].to(device))
        chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float32)
