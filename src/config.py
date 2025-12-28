from pathlib import Path
from types import SimpleNamespace

import yaml


def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    return d


def load(path: str | Path):
    with open(path) as f:
        return _ns(yaml.safe_load(f))
