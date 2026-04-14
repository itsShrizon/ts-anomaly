"""NASA C-MAPSS turbofan. space-separated, no header."""
import numpy as np
import pandas as pd

from .loader import data_root

COLS = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"sensor{i}" for i in range(1, 22)]


def load_fd(fd: str = "FD001", split: str = "train") -> pd.DataFrame:
    p = data_root() / "raw" / "turbofan" / f"{split}_{fd}.txt"
    df = pd.read_csv(p, sep=r"\s+", header=None, names=COLS, engine="python")
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=125).astype(np.int32)
    return df
