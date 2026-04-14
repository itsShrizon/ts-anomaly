#!/usr/bin/env python
"""tiny cartesian sweep. no hyperopt, no tuners, just a grid + mlflow runs."""
import argparse
import copy
import itertools
import subprocess
import sys

import yaml


def _set(d, dotted: str, value):
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/default.yaml")
    ap.add_argument("--grid", default="configs/sweep.yaml")
    args = ap.parse_args()

    with open(args.base) as f:
        base = yaml.safe_load(f)
    with open(args.grid) as f:
        grid = yaml.safe_load(f)["grid"]
    keys = list(grid.keys())

    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo, strict=True):
            _set(cfg, k, v)
        name = "__".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo, strict=True))
        out = f"/tmp/{name}.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f)
        subprocess.run([sys.executable, "scripts/train.py", "--config", out, "--run", name], check=True)


if __name__ == "__main__":
    main()
