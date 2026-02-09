"""wire dataset + loaders from a config."""
import numpy as np
from torch.utils.data import DataLoader

from .dataset import WindowedSeries
from .scaling import Standardizer
from .skab import SENSOR_COLS, load_skab


def _split(df, x_cols, y_col="anomaly"):
    return df[x_cols].to_numpy(dtype=np.float32), df[y_col].to_numpy(dtype=np.float32)


def build_loaders(cfg):
    df_tr = load_skab("train")
    try:
        df_va = load_skab("valid")
    except FileNotFoundError:
        # fallback: last 20% of train
        cut = int(len(df_tr) * 0.8)
        df_va = df_tr.iloc[cut:].reset_index(drop=True)
        df_tr = df_tr.iloc[:cut].reset_index(drop=True)

    xtr, ytr = _split(df_tr, SENSOR_COLS)
    xva, yva = _split(df_va, SENSOR_COLS)

    sc = Standardizer.fit(xtr)
    xtr, xva = sc.transform(xtr), sc.transform(xva)

    ds_tr = WindowedSeries(xtr, ytr, cfg.data.window, cfg.data.stride)
    ds_va = WindowedSeries(xva, yva, cfg.data.window, cfg.data.stride)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.data.batch_size, shuffle=True,
                       num_workers=cfg.data.num_workers, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.data.batch_size, shuffle=False,
                       num_workers=cfg.data.num_workers)
    return dl_tr, dl_va, xtr.shape[1]
