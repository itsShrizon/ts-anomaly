import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    return float(f1[i]), float(t[i])


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def f1_at(y_true: np.ndarray, scores: np.ndarray, thr: float) -> float:
    return float(f1_score(y_true, (scores >= thr).astype(int), zero_division=0))
