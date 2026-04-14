"""SKAB loader. one csv per run, 'anomaly' column is the label.

real repo layout after `scripts/fetch_data.sh`:
  data/raw/skab/data/{anomaly-free,valve1,valve2,other}/*.csv
"""
import pandas as pd

from .loader import data_root

SENSOR_COLS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current",
               "Pressure", "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]


_SPLITS = {
    "train": ["anomaly-free", "valve1"],
    "valid": ["valve2"],
    "test": ["other"],
    "all": ["anomaly-free", "valve1", "valve2", "other"],
}


def load_skab(split: str = "train") -> pd.DataFrame:
    base = data_root() / "raw" / "skab" / "data"
    dirs = _SPLITS.get(split, [split])
    frames = []
    for d in dirs:
        for p in sorted((base / d).glob("*.csv")):
            f = pd.read_csv(p, sep=";", parse_dates=["datetime"])
            if "anomaly" not in f.columns:
                f["anomaly"] = 0.0
            frames.append(f)
    if not frames:
        raise FileNotFoundError(f"no skab csvs under {base} for split={split}")
    df = pd.concat(frames, ignore_index=True)
    df[SENSOR_COLS] = df[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=SENSOR_COLS + ["anomaly"]).reset_index(drop=True)
    return df
