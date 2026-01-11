"""rolling median post-process to kill single-frame flickers."""
import numpy as np


def median_smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i + k])
    return out
