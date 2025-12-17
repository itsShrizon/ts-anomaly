import numpy as np
import torch
from torch.utils.data import Dataset

from .windows import sliding, windowed_labels


class WindowedSeries(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, win: int, stride: int = 1):
        self.x = sliding(x.astype(np.float32), win, stride)
        self.y = windowed_labels(y.astype(np.float32), win, stride)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])
