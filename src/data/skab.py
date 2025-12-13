"""SKAB loader. one csv per run, 'anomaly' column is the label."""
from pathlib import Path
import pandas as pd

from .loader import data_root


SENSOR_COLS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current",
               "Pressure", "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]


def load_skab(split: str = "train") -> pd.DataFrame:
    root = data_root() / "raw" / "skab" / split
    frames = [pd.read_csv(p, sep=";", parse_dates=["datetime"]) for p in sorted(root.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no skab csvs at {root}")
    return pd.concat(frames, ignore_index=True)
