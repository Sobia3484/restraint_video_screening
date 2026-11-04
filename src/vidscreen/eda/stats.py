"""Small, dependency-light statistics for the platform comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PLATFORMS = ("youtube", "tiktok")


def describe(x) -> dict:
    x = pd.Series(x).dropna().astype(float)
    if x.empty:
        return {"n": 0}
    q = x.quantile([0.01, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {"n": int(len(x)), "mean": x.mean(), "std": x.std(), "min": x.min(), "p01": q[0.01], "q1": q[0.25], "median": q[0.5], "q3": q[0.75],
            "p90": q[0.9], "p99": q[0.99], "max": x.max()}


def cliffs_delta(x, y) -> float:
    """P(x > y) - P(x < y); positive means x tends to be larger. Computed from sorted ranks (fast)."""
    x, y = np.asarray(x, dtype=float), np.sort(np.asarray(y, dtype=float))
    less = np.searchsorted(y, x, side="left")
    greater = len(y) - np.searchsorted(y, x, side="right")
    return float((less - greater).sum() / (len(x) * len(y)))


def magnitude(d: float) -> str:
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def compare_two(x, y, n_boot: int = 2000, seed: int = 42) -> dict:
    """Mann-Whitney U test, Cliff's delta and median difference with bootstrap 95% confidence intervals (x = first group)."""
    x = pd.Series(x).dropna().astype(float).to_numpy()
    y = pd.Series(y).dropna().astype(float).to_numpy()
    if len(x) < 2 or len(y) < 2:
        return {"n_x": len(x), "n_y": len(y)}
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    rng = np.random.default_rng(seed)
    deltas, diffs = [], []
    for _ in range(n_boot):
        bx, by = rng.choice(x, len(x)), rng.choice(y, len(y))
        deltas.append(cliffs_delta(bx, by))
        diffs.append(np.median(bx) - np.median(by))
    d = cliffs_delta(x, y)
    return {"n_x": len(x), "n_y": len(y), "median_x": float(np.median(x)), "median_y": float(np.median(y)),
            "median_diff": float(np.median(x) - np.median(y)), "median_diff_ci_low": float(np.quantile(diffs, 0.025)), "median_diff_ci_high": float(np.quantile(diffs, 0.975)),
            "u_statistic": float(u.statistic), "p_value": float(u.pvalue), "cliffs_delta": d,
            "delta_ci_low": float(np.quantile(deltas, 0.025)), "delta_ci_high": float(np.quantile(deltas, 0.975)), "magnitude": magnitude(d)}


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def gini(values) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0 or v.sum() == 0:
        return float("nan")
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))
