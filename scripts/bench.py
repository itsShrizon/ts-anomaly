#!/usr/bin/env python
"""latency benchmark. target: <50ms per window on pi-class."""
import argparse
import time

import numpy as np

from src.infer.ort_session import OrtScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    sc = OrtScorer(args.model, threads=args.threads)
    x = np.random.randn(args.batch, args.window, args.channels).astype(np.float32)

    for _ in range(args.warmup):
        sc(x)

    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sc(x)
        ts.append((time.perf_counter() - t0) * 1000)

    ts = np.array(ts)
    per_win = ts / args.batch
    print(f"batch={args.batch}  threads={args.threads}")
    print(f"  total:    p50={np.percentile(ts, 50):.2f}ms  p95={np.percentile(ts, 95):.2f}ms")
    print(f"  per-win:  p50={np.percentile(per_win, 50):.2f}ms  p95={np.percentile(per_win, 95):.2f}ms")


if __name__ == "__main__":
    main()
