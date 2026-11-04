"""Inter-annotator agreement and the adjudication list."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

LABELS = ["Relevant", "Irrelevant", "Unsure"]


def cohen_kappa(a: List[str], b: List[str], labels: List[str] = None) -> float:
    """Cohen's kappa for two raters (unweighted). NaN when undefined."""
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    labels = labels or sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(axis=1) * m.sum(axis=0)).sum() / n**2)
    if pe == 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return float((po - pe) / (1 - pe))


def _check(df: pd.DataFrame, who: str) -> None:
    bad = df[~df["label"].isin(LABELS)]
    if len(bad):
        raise ValueError(
            f"{who}: {len(bad)} rows have an empty or invalid label (e.g. item {bad.iloc[0]['item_id']}). "
            f"Allowed: {LABELS}"
        )


def compare_annotators(a: pd.DataFrame, b: pd.DataFrame, platforms: pd.Series = None) -> Dict:
    """Agreement on the items both annotators labeled. ``platforms``: Series item_id -> platform."""
    _check(a, "annotator A")
    _check(b, "annotator B")
    m = a.merge(b, on="item_id", suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("No common item_id between the two annotators.")
    la, lb = m["label_a"].tolist(), m["label_b"].tolist()
    rep = {
        "n_items": len(m),
        "percent_agreement": round(100 * float(np.mean(np.array(la) == np.array(lb))), 1),
        "kappa_3class": round(cohen_kappa(la, lb, LABELS), 3),
        "kappa_relevant_vs_rest": round(
            cohen_kappa(["R" if x == "Relevant" else "N" for x in la], ["R" if x == "Relevant" else "N" for x in lb]), 3
        ),
        "confusion_A_rows_B_cols": pd.crosstab(
            pd.Categorical(m["label_a"], LABELS), pd.Categorical(m["label_b"], LABELS), dropna=False
        ).values.tolist(),
        "label_order": LABELS,
        "unsure_A": int((m["label_a"] == "Unsure").sum()),
        "unsure_B": int((m["label_b"] == "Unsure").sum()),
    }
    if platforms is not None:
        m["platform"] = m["item_id"].map(platforms)
        rep["by_platform"] = {
            p: {"n": len(g), "kappa_3class": round(cohen_kappa(g["label_a"].tolist(), g["label_b"].tolist(), LABELS), 3)}
            for p, g in m.groupby("platform")
        }
    return rep


def build_adjudication_list(key: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame = None) -> pd.DataFrame:
    """Items needing a third adjudicator: A/B disagreements and anything labeled Unsure."""
    _check(a, "annotator A")
    m = key[["item_id", "platform", "url", "title"]].merge(
        a.rename(columns={c: f"{c}_a" for c in a.columns if c != "item_id"}), on="item_id", how="left"
    )
    if b is not None and len(b):
        _check(b, "annotator B")
        m = m.merge(b.rename(columns={c: f"{c}_b" for c in b.columns if c != "item_id"}), on="item_id", how="left")
    else:
        m["label_b"] = ""
    for c in ("label_b", "note_b", "unsure_reason_b"):
        if c not in m:
            m[c] = ""
    m = m.fillna("")
    disagree = (m["label_b"] != "") & (m["label_a"] != m["label_b"])
    unsure = (m["label_a"] == "Unsure") | (m["label_b"] == "Unsure")
    m["reason"] = np.where(disagree, "disagreement", np.where(unsure, "unsure", ""))
    out = m[m["reason"] != ""].copy()
    out["final_label"] = ""
    out["adjudicator"] = ""
    out["adjudication_note"] = ""
    cols = ["item_id", "platform", "url", "title", "reason", "label_a", "label_b", "note_a", "note_b",
            "unsure_reason_a", "unsure_reason_b", "final_label", "adjudicator", "adjudication_note"]
    return out[cols].reset_index(drop=True)
