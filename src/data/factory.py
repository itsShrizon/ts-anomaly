"""wire dataset + loaders from a config. skab only for now."""
import numpy as np
from torch.utils.data import DataLoader

from .dataset import WindowedSeries
from .scaling import Standardizer
from .skab import SENSOR_COLS, load_skab


def build_loaders(cfg):
    df_tr = load_skab("train")
    df_va = load_skab("valid") if (cfg.data.dataset == "skab") else load_skab("train")

    xtr = df_tr[SENSOR_COLS].to_numpy(dtype=np.float32)
    ytr = df_tr["anomaly"].to_numpy(dtype=np.float32)
    xva = df_va[SENSOR_COLS].to_numpy(dtype=np.float32)
    yva = df_va["anomaly"].to_numpy(dtype=np.float32)

    sc = Standardizer.fit(xtr)
    xtr, xva = sc.transform(xtr), sc.transform(xva)

    ds_tr = WindowedSeries(xtr, ytr, cfg.data.window, cfg.data.stride)
    ds_va = WindowedSeries(xva, yva, cfg.data.window, cfg.data.stride)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    dl_va = DataLoader(ds_va, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    return dl_tr, dl_va, xtr.shape[1]
