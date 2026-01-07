"""thin mlflow wrapper. no-op if mlflow not installed."""
from __future__ import annotations

try:
    import mlflow
    _ON = True
except ImportError:
    _ON = False


def start(run_name: str, params: dict | None = None):
    if not _ON:
        return None
    mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log(metrics: dict, step: int | None = None):
    if not _ON:
        return
    mlflow.log_metrics(metrics, step=step)


def artifact(path: str):
    if _ON:
        mlflow.log_artifact(path)


def end():
    if _ON:
        mlflow.end_run()
