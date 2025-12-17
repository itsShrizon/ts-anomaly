import numpy as np
from src.data.windows import sliding, windowed_labels


def test_sliding_shape():
    x = np.arange(20).reshape(10, 2)
    w = sliding(x, win=4, stride=2)
    assert w.shape == (4, 4, 2)


def test_labels_align():
    y = np.arange(10)
    out = windowed_labels(y, win=4, stride=2)
    assert out.tolist() == [3, 5, 7, 9]
