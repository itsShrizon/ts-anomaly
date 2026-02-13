"""wire dataset + loaders from a config. supports skab + turbofan."""
import numpy as np
from torch.utils.data import DataLoader

from .dataset import WindowedSeries
from .scaling import Standardizer
from .skab import SENSOR_COLS, load_skab
from .turbofan import COLS as TF_COLS
from .turbofan import add_rul, load_fd


def _skab():
    df_tr = load_skab("train")
    try:
        df_va = load_skab("valid")
    except FileNotFoundError:
        cut = int(len(df_tr) * 0.8)
        df_va = df_tr.iloc[cut:].reset_index(drop=True)
        df_tr = df_tr.iloc[:cut].reset_index(drop=True)
    x_cols, y_col = SENSOR_COLS, "anomaly"
    return df_tr, df_va, x_cols, y_col


def _turbofan():
    df_tr = add_rul(load_fd("FD001", "train"))
    cut = int(df_tr["unit"].max() * 0.8)
    tr = df_tr[df_tr["unit"] <= cut]
    va = df_tr[df_tr["unit"] > cut]
    # label: RUL < 30 cycles == soon-to-fail
    x_cols = [c for c in TF_COLS if c.startswith("sensor")]
    tr = tr.assign(anomaly=(tr["rul"] < 30).astype(np.float32))
    va = va.assign(anomaly=(va["rul"] < 30).astype(np.float32))
    return tr.reset_index(drop=True), va.reset_index(drop=True), x_cols, "anomaly"


def build_loaders(cfg):
    if cfg.data.dataset == "skab":
        df_tr, df_va, x_cols, y_col = _skab()
    elif cfg.data.dataset == "turbofan":
        df_tr, df_va, x_cols, y_col = _turbofan()
    else:
        raise ValueError(cfg.data.dataset)

    xtr = df_tr[x_cols].to_numpy(dtype=np.float32)
    ytr = df_tr[y_col].to_numpy(dtype=np.float32)
    xva = df_va[x_cols].to_numpy(dtype=np.float32)
    yva = df_va[y_col].to_numpy(dtype=np.float32)

    sc = Standardizer.fit(xtr)
    xtr, xva = sc.transform(xtr), sc.transform(xva)

    ds_tr = WindowedSeries(xtr, ytr, cfg.data.window, cfg.data.stride)
    ds_va = WindowedSeries(xva, yva, cfg.data.window, cfg.data.stride)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.data.batch_size, shuffle=True,
                       num_workers=cfg.data.num_workers, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.data.batch_size, shuffle=False,
                       num_workers=cfg.data.num_workers)
    return dl_tr, dl_va, xtr.shape[1]
