#!/usr/bin/env python
import argparse
import json

import numpy as np

from src.eval.report import report
from src.infer.ort_session import OrtScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()

    x = np.load(args.x).astype(np.float32)
    y = np.load(args.y).astype(np.int32)
    sc = OrtScorer(args.model)
    probs = sc(x)
    out = report(y, probs, args.thr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
