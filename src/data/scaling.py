"""per-channel standardization. fit on train, freeze for eval."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return cls(mu=mu, sd=sd)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd
