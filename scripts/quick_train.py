#!/usr/bin/env python
"""short training run + onnx export + int8 quant — populates artifacts/ end-to-end."""
from pathlib import Path

import numpy as np
import torch

from src.config import load as load_cfg
from src.data.factory import build_loaders
from src.export.onnx_export import export as to_onnx
from src.export.quantize import quantize
from src.models.hybrid import HybridAnomaly
from src.models.loss import FocalBCE
from src.train.checkpoint import save
from src.train.loop import evaluate, train_one_epoch


def main():
    cfg = load_cfg("configs/default.yaml")
    cfg.train.epochs = 3  # short demo run
    cfg.data.batch_size = 64
    cfg.data.num_workers = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_tr, dl_va, in_dim = build_loaders(cfg)

    model = HybridAnomaly(in_dim, cfg.model.hidden, cfg.model.rnn_layers,
                          cfg.model.heads, cfg.model.attn_layers, cfg.model.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = FocalBCE(cfg.loss.alpha, cfg.loss.gamma)

    for epoch in range(cfg.train.epochs):
        tr = train_one_epoch(model, dl_tr, opt, loss_fn, device, cfg.train.grad_clip)
        va = evaluate(model, dl_va, loss_fn, device)
        print(f"epoch {epoch}  train={tr:.4f}  val={va:.4f}")

    ckpt = "checkpoints/last.pt"
    save(model, opt, cfg.train.epochs, ckpt)
    print(f"ckpt: {Path(ckpt).stat().st_size/1024:.1f} KB")

    to_onnx(model, in_dim, cfg.data.window, "artifacts/model.onnx")
    print(f"fp32 onnx: {Path('artifacts/model.onnx').stat().st_size/1024:.1f} KB")

    samples = []
    for x, _ in dl_tr:
        samples.append(x.numpy())
        if len(samples) >= 32:
            break
    quantize("artifacts/model.onnx", "artifacts/model.int8.onnx", samples)
    print(f"int8 onnx: {Path('artifacts/model.int8.onnx').stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
