import numpy as np
import torch


@torch.no_grad()
def score(model, x: np.ndarray, device: str = "cpu", bs: int = 256) -> np.ndarray:
    model.eval().to(device)
    xt = torch.from_numpy(x.astype(np.float32))
    out = []
    for i in range(0, len(xt), bs):
        o = torch.sigmoid(model(xt[i:i + bs].to(device))).cpu().numpy()
        out.append(o)
    return np.concatenate(out) if out else np.empty((0,), dtype=np.float32)
