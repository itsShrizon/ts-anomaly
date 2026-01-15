#!/usr/bin/env python
import argparse

import numpy as np
import torch

from src.config import load as load_cfg
from src.data.factory import build_loaders
from src.export.onnx_export import export as to_onnx
from src.export.quantize import quantize
from src.models.hybrid import HybridAnomaly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fp32", default="artifacts/model.onnx")
    ap.add_argument("--int8", default="artifacts/model.int8.onnx")
    ap.add_argument("--calib", type=int, default=200)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    dl_tr, _, in_dim = build_loaders(cfg)

    model = HybridAnomaly(in_dim, cfg.model.hidden, cfg.model.rnn_layers,
                          cfg.model.heads, cfg.model.attn_layers, cfg.model.dropout)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])

    to_onnx(model, in_dim, cfg.data.window, args.fp32)

    samples: list[np.ndarray] = []
    for x, _ in dl_tr:
        samples.append(x.numpy())
        if len(samples) >= args.calib:
            break
    quantize(args.fp32, args.int8, samples)
    print(f"wrote {args.int8}")


if __name__ == "__main__":
    main()
