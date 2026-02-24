"""one-shot eval report. prints + returns a dict."""
import numpy as np

from .metrics import auroc, best_f1, f1_at


def report(y: np.ndarray, scores: np.ndarray, thr: float | None = None) -> dict:
    f1_star, thr_star = best_f1(y, scores)
    chosen = thr if thr is not None else thr_star
    return {
        "auroc": auroc(y, scores),
        "best_f1": f1_star,
        "best_threshold": thr_star,
        "f1_at_chosen": f1_at(y, scores, chosen),
        "threshold_used": chosen,
    }
