from __future__ import annotations

from pathlib import Path

import warnings

import matplotlib

warnings.filterwarnings("ignore", message="Glyph .* missing from font")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"youtube": "#C0392B", "tiktok": "#1F3864"}
NAMES = {"youtube": "YouTube", "tiktok": "TikTok"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def hist_by_platform(df, col, path, title, xlabel, log_x=False, bins=40) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vals = df[col].dropna()
    vals = vals[vals > 0] if log_x else vals
    edges = np.geomspace(vals.min(), vals.max(), bins) if log_x else np.linspace(vals.min(), vals.max(), bins)
    for p in ("youtube", "tiktok"):
        v = df.loc[df["platform"] == p, col].dropna()
        v = v[v > 0] if log_x else v
        ax.hist(v, bins=edges, alpha=0.6, color=COLORS[p], label=f"{NAMES[p]} (n={len(v)})")
        if len(v):
            ax.axvline(v.median(), color=COLORS[p], linestyle="--", linewidth=1)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Videos", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return save(fig, path)


def box_by_platform(df, cols: dict, path, title, log_y=False) -> str:
    """cols: {label: column}; one box per platform per label."""
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for ax, (label, col) in zip(axes[0], cols.items()):
        data = [df.loc[df["platform"] == p, col].dropna() for p in ("youtube", "tiktok")]
        if log_y:
            data = [d[d > 0] for d in data]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True, flierprops={"markersize": 2, "alpha": 0.4}, medianprops={"color": "white", "linewidth": 1.5})
        for patch, p in zip(bp["boxes"], ("youtube", "tiktok")):
            patch.set_facecolor(COLORS[p])
        ax.set_xticks([1, 2]); ax.set_xticklabels([f"YouTube\n(n={len(data[0])})", f"TikTok\n(n={len(data[1])})"], fontsize=8)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=9)
        _style(ax)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return save(fig, path)


def stacked_share(table: pd.DataFrame, path, title, xlabel="Share of videos (%)") -> str:
    """table: rows = platforms, columns = categories, values = counts."""
    share = table.div(table.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7, 2.6 + 0.2 * len(share.columns)))
    cmap = plt.get_cmap("Blues")
    left = np.zeros(len(share))
    for i, c in enumerate(share.columns):
        ax.barh([NAMES.get(p, p) for p in share.index], share[c], left=left, color=cmap(0.25 + 0.7 * i / max(1, len(share.columns) - 1)), label=str(c))
        for j, v in enumerate(share[c]):
            if v >= 6:
                ax.text(left[j] + v / 2, j, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white" if i > len(share.columns) / 2 else "black")
        left += share[c].to_numpy()
    ax.set_xlim(0, 100); ax.set_xlabel(xlabel, fontsize=9); ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=min(5, len(share.columns)), loc="upper center", bbox_to_anchor=(0.5, -0.25))
    _style(ax)
    return save(fig, path)


def bar_counts(series: pd.Series, path, title, xlabel="Videos", color="#1F3864", top=None) -> str:
    s = series.sort_values(ascending=True)
    if top:
        s = s.tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.2))
    ax.barh(s.index.astype(str), s.values, color=color)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=7)
    ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel(xlabel, fontsize=9)
    _style(ax)
    return save(fig, path)


def ecdf_by_platform(df, col, path, title, xlabel, log_x=True) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for p in ("youtube", "tiktok"):
        v = np.sort(df.loc[df["platform"] == p, col].dropna().to_numpy())
        v = v[v > 0] if log_x else v
        if len(v):
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[p], label=NAMES[p])
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("Share of videos at or below", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return save(fig, path)


GROUP_COLORS = {"relevant": "#1F3864", "audit": "#E0A800", "irrelevant": "#8C8C8C"}


def box_by_group(df, col, group_col, order, path, title, log_y=False, ylabel="") -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    data = [df.loc[df[group_col] == g, col].dropna() for g in order]
    if log_y:
        data = [x[x > 0] for x in data]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, flierprops={"markersize": 2, "alpha": 0.4}, medianprops={"color": "white", "linewidth": 1.5})
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(GROUP_COLORS.get(g, "#555555"))
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([f"{g}\n(n={len(x)})" for g, x in zip(order, data)], fontsize=8)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    return save(fig, path)


def hist_by_group(df, col, group_col, order, path, title, xlabel, bins=30) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    edges = np.linspace(float(df[col].min()), float(df[col].max()), bins)
    for g in order:
        v = df.loc[df[group_col] == g, col].dropna()
        ax.hist(v, bins=edges, alpha=0.65, color=GROUP_COLORS.get(g, "#555555"), label=f"{g} (n={len(v)})")
    ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("Videos", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return save(fig, path)


def grouped_bars(table: pd.DataFrame, path, title, xlabel="Share of videos (%)") -> str:
    """table: rows = categories, columns = groups (values in percent). Horizontal grouped bars."""
    n, k = len(table), len(table.columns)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * n * max(1, k * 0.6) + 1.4))
    h = 0.8 / k
    for j, c in enumerate(table.columns):
        ax.barh(np.arange(n) + j * h, table[c].to_numpy(), height=h, color=COLORS.get(c, GROUP_COLORS.get(c, "#555555")), label=NAMES.get(c, c))
    ax.set_yticks(np.arange(n) + h * (k - 1) / 2); ax.set_yticklabels(table.index, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel(xlabel, fontsize=9); ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return save(fig, path)
