"""sliding windows over multivariate series."""
import numpy as np


def sliding(x: np.ndarray, win: int, stride: int = 1) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("expected (T, C)")
    t = x.shape[0]
    if t < win:
        return np.empty((0, win, x.shape[1]), dtype=x.dtype)
    n = (t - win) // stride + 1
    idx = np.arange(win)[None, :] + (np.arange(n) * stride)[:, None]
    return x[idx]


def windowed_labels(y: np.ndarray, win: int, stride: int = 1) -> np.ndarray:
    """label at end of window, same count as sliding()."""
    t = len(y)
    if t < win:
        return np.empty((0,), dtype=y.dtype)
    n = (t - win) // stride + 1
    ends = (np.arange(n) * stride) + (win - 1)
    return y[ends]
