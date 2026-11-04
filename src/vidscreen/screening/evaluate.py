"""Metrics for the binary task Relevant (1) vs Irrelevant (0). Optional sample weights."""
from __future__ import annotations

import numpy as np


def metrics(y_true, y_pred, weights=None) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    tp = w[(y_true == 1) & (y_pred == 1)].sum()
    fp = w[(y_true == 0) & (y_pred == 1)].sum()
    fn = w[(y_true == 1) & (y_pred == 0)].sum()
    tn = w[(y_true == 0) & (y_pred == 0)].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"precision": precision, "recall": recall, "f1": f1, "accuracy": (tp + tn) / w.sum() if w.sum() else 0.0, "n": int(len(y_true))}
    if weights is None:
        out.update({"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def bootstrap_ci(y_true, y_pred, key: str = "precision", n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(metrics(y_true[idx], y_pred[idx])[key])
    return round(float(np.quantile(vals, alpha / 2)), 3), round(float(np.quantile(vals, 1 - alpha / 2)), 3)


def recall_at_precision(y, p, target: float, weights=None, min_pred: int = 5) -> float:
    """Highest recall reachable while precision >= target (NaN if the target is never reached)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    best = float("nan")
    for t in np.unique(np.round(p, 3)):
        pred = p >= t
        if pred.sum() < min_pred:
            continue
        m = metrics(y, pred, weights)
        if m["precision"] >= target and (np.isnan(best) or m["recall"] > best):
            best = m["recall"]
    return best