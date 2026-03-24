"""rolling median post-process to kill single-frame flickers."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def median_smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.median(sliding_window_view(xp, k), axis=1)
