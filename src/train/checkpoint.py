from pathlib import Path

import torch


def save(model, opt, epoch: int, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, path)


def load(model, path: str | Path, opt=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return ckpt.get("epoch", 0)
