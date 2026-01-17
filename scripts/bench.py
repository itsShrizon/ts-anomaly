#!/usr/bin/env python
"""latency benchmark on a cpu-only box. target: <50ms per window on pi-class."""
import argparse
import time

import numpy as np

from src.infer.ort_session import OrtScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    sc = OrtScorer(args.model)
    x = np.random.randn(1, args.window, args.channels).astype(np.float32)

    for _ in range(args.warmup):
        sc(x)

    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sc(x)
        ts.append((time.perf_counter() - t0) * 1000)

    ts = np.array(ts)
    print(f"p50={np.percentile(ts, 50):.2f}ms  p95={np.percentile(ts, 95):.2f}ms  mean={ts.mean():.2f}ms")


if __name__ == "__main__":
    main()
