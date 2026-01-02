#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load as load_cfg
from src.models.hybrid import HybridAnomaly
from src.models.loss import FocalBCE
from src.train.checkpoint import save
from src.train.early_stop import EarlyStop
from src.train.loop import train_one_epoch, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="checkpoints/last.pt")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets are wired in by the caller via a factory in real runs; kept minimal here.
    from src.data.factory import build_loaders
    train_dl, val_dl, in_dim = build_loaders(cfg)

    model = HybridAnomaly(in_dim, cfg.model.hidden, cfg.model.rnn_layers,
                          cfg.model.heads, cfg.model.attn_layers, cfg.model.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = FocalBCE(cfg.loss.alpha, cfg.loss.gamma)
    stopper = EarlyStop(cfg.train.patience)

    for epoch in range(cfg.train.epochs):
        tr = train_one_epoch(model, train_dl, opt, loss_fn, device, cfg.train.grad_clip)
        va = evaluate(model, val_dl, loss_fn, device)
        print(f"epoch {epoch:03d}  train {tr:.4f}  val {va:.4f}")
        save(model, opt, epoch, args.out)
        if stopper.step(va):
            print("early stop")
            break


if __name__ == "__main__":
    main()
