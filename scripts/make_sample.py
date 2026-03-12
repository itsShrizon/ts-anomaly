#!/usr/bin/env python
"""generate a tiny SKAB-shaped csv for smoke tests."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.skab import SENSOR_COLS


def main():
    rng = np.random.default_rng(42)
    n = 2_000
    t = pd.date_range("2026-01-01", periods=n, freq="s")
    data = {c: rng.normal(size=n) for c in SENSOR_COLS}
    # inject a few anomaly spans
    y = np.zeros(n, dtype=int)
    for start in (400, 900, 1600):
        y[start:start + 50] = 1
        for c in SENSOR_COLS[:3]:
            data[c][start:start + 50] += rng.normal(3, 0.5, 50)
    df = pd.DataFrame({"datetime": t, **data, "anomaly": y, "changepoint": 0})
    out = Path("data/sample/sample.csv")
    df.to_csv(out, sep=";", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
