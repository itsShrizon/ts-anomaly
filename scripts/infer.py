#!/usr/bin/env python
import argparse

import numpy as np

from src.infer.ort_session import OrtScorer
from src.infer.smooth import median_smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True, help="pre-windowed array (N, T, C) float32")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--out", default="predictions.npy")
    args = ap.parse_args()

    x = np.load(args.npy).astype(np.float32)
    sc = OrtScorer(args.model)
    probs = sc(x)
    probs = median_smooth(probs, args.smooth)
    y = (probs >= args.thr).astype(np.int8)
    np.save(args.out, y)
    print(f"wrote {args.out}  pos={int(y.sum())}/{len(y)}")


if __name__ == "__main__":
    main()
