"""export hybrid to onnx w/ dynamic batch + time axes."""
from pathlib import Path

import torch


def export(model, in_dim: int, win: int, out: str | Path, opset: int = 17) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.randn(1, win, in_dim, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, out.as_posix(),
        input_names=["x"], output_names=["score"],
        dynamic_axes={"x": {0: "batch", 1: "time"}, "score": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return out
