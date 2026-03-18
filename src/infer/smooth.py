"""rolling median post-process to kill single-frame flickers."""
import numpy as np


def median_smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1  # force odd window; center is well-defined
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    # stride trick: (N, k) view → median along axis=1
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(xp, k), axis=1)
