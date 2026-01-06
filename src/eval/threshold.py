"""pick threshold once on validation, freeze for test."""
import json
from pathlib import Path

import numpy as np

from .metrics import best_f1


def tune_and_save(y: np.ndarray, scores: np.ndarray, out: str | Path) -> float:
    f1, thr = best_f1(y, scores)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"threshold": thr, "val_f1": f1}, f, indent=2)
    return thr


def load(path: str | Path) -> float:
    with open(path) as f:
        return float(json.load(f)["threshold"])
