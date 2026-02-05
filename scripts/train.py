#!/usr/bin/env python
import argparse

import torch

from src import track
from src.config import load as load_cfg
from src.data.factory import build_loaders
from src.device import pick as pick_device
from src.log_setup import setup as setup_log
from src.models.hybrid import HybridAnomaly
from src.models.loss import FocalBCE
from src.train.checkpoint import save
from src.train.early_stop import EarlyStop
from src.train.loop import evaluate, train_one_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="checkpoints/last.pt")
    ap.add_argument("--run", default="default")
    args = ap.parse_args()

    log = setup_log()
    cfg = load_cfg(args.config)
    device = pick_device()
    log.info("device=%s", device)

    track.start(args.run, {"lr": cfg.train.lr, "hidden": cfg.model.hidden, "win": cfg.data.window})

    dl_tr, dl_va, in_dim = build_loaders(cfg)
    model = HybridAnomaly(in_dim, cfg.model.hidden, cfg.model.rnn_layers,
                          cfg.model.heads, cfg.model.attn_layers, cfg.model.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = FocalBCE(cfg.loss.alpha, cfg.loss.gamma)
    stopper = EarlyStop(cfg.train.patience)

    for epoch in range(cfg.train.epochs):
        tr = train_one_epoch(model, dl_tr, opt, loss_fn, device, cfg.train.grad_clip)
        va = evaluate(model, dl_va, loss_fn, device)
        track.log({"train_loss": tr, "val_loss": va}, step=epoch)
        log.info("epoch %03d  train %.4f  val %.4f", epoch, tr, va)
        save(model, opt, epoch, args.out)
        if stopper.step(va):
            log.info("early stop @ %d", epoch)
            break

    track.artifact(args.out)
    track.end()


if __name__ == "__main__":
    main()
