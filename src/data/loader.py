"""Dataset loaders."""
from pathlib import Path


def data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"
