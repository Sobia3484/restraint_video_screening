"""Sampling for human annotation.

* ``assign_primary_subtopic``: a video found by several queries gets the subtopic of the
  query where it ranked best (ties: the first one listed).
* ``stratified_sample``: random sample per platform, stratified by subtopic, with a floor per
  stratum so that small subtopics (dental, wound care, neonatal) are still represented.
  Each sampled video gets ``sample_weight = stratum_size / n_sampled_in_stratum`` so that
  performance on the full dataset can be estimated later.
* ``diverse_sample``: pilot sample. Greedy selection that spreads the picks over platform,
  subtopic, transcript presence, duration bucket and language, so the pilot covers hard cases.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

import numpy as np
import pandas as pd


def assign_primary_subtopic(master: pd.DataFrame, query_map: Optional[pd.DataFrame]) -> pd.Series:
    """Return a Series (index = master index) with one subtopic per video."""
    first = master["query_subtopics"].fillna("").astype(str).str.split("|").str[0].replace("", "unknown")
    if query_map is None or query_map.empty:
        return first
    qm = query_map.copy()
    qm["query_rank"] = pd.to_numeric(qm["query_rank"], errors="coerce").fillna(10**6)
    qm = qm.sort_values(["platform", "video_id", "query_rank"], kind="stable")
    best = qm.drop_duplicates(["platform", "video_id"])[["platform", "video_id", "subtopic"]]
    merged = master[["platform", "video_id"]].merge(best, on=["platform", "video_id"], how="left")
    merged.index = master.index
    return merged["subtopic"].where(merged["subtopic"].notna() & (merged["subtopic"] != ""), first)


def allocate(sizes: Dict[str, int], n: int, min_each: int = 8) -> Dict[str, int]:
    """Split ``n`` draws over strata: a floor per stratum, the rest proportional to size."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    base = {k: min(min_each, s) for k, s in sizes.items()}
    if sum(base.values()) > n:  # floor too large for n: plain proportional
        base = {k: 0 for k in sizes}
    alloc = dict(base)
    remaining = n - sum(alloc.values())
    while remaining > 0:
        room = {k: sizes[k] - alloc[k] for k in sizes if sizes[k] - alloc[k] > 0}
        if not room:
            break
        room_total = sum(room.values())
        shares = {k: remaining * r / room_total for k, r in room.items()}
        add = {k: min(int(np.floor(v)), room[k]) for k, v in shares.items()}
        gave = sum(add.values())
        if gave == 0:  # hand out the leftover one by one, largest remainder first
            for k in sorted(shares, key=lambda k: shares[k] - np.floor(shares[k]), reverse=True)[:remaining]:
                add[k] = 1
            gave = sum(add.values())
        for k, v in add.items():
            alloc[k] += v
        remaining -= gave
    return alloc


def stratified_sample(
    df: pd.DataFrame,
    n_per_platform: int,
    seed: int = 42,
    strata_col: str = "primary_subtopic",
    min_each: int = 8,
) -> pd.DataFrame:
    """Random sample, ``n_per_platform`` per platform, stratified by ``strata_col``."""
    rng = np.random.default_rng(seed)
    parts = []
    for platform, g in df.groupby("platform", sort=True):
        sizes = g.groupby(strata_col).size().to_dict()
        alloc = allocate(sizes, n_per_platform, min_each)
        for stratum, k in alloc.items():
            if k <= 0:
                continue
            members = g[g[strata_col] == stratum]
            pick = members.iloc[rng.choice(len(members), size=k, replace=False)].copy()
            pick["stratum"] = f"{platform}|{stratum}"
            pick["stratum_size"] = len(members)
            pick["sample_weight"] = len(members) / k
            parts.append(pick)
    return pd.concat(parts).sort_index() if parts else df.iloc[0:0].copy()


def _duration_bucket(sec) -> str:
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return "unknown"
    return "lt1min" if s < 60 else "1to5min" if s <= 300 else "gt5min"


def _features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["platform"] = df["platform"]
    out["subtopic"] = df["primary_subtopic"]
    out["transcript"] = df["has_transcript"].astype(str).str.lower().isin(["true", "1", "yes"]).map(
        {True: "tr", False: "no_tr"}
    )
    out["duration"] = df["duration_seconds"].map(_duration_bucket)
    lang = df["language_clean"].fillna("unknown").astype(str).str.lower()
    out["lang"] = np.where(lang.isin(["en", "english"]), "en", np.where(lang == "unknown", "unknown", "other"))
    return out


def diverse_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Pilot sample: half per platform, greedy to cover subtopic/transcript/duration/language."""
    rng = np.random.default_rng(seed)
    feats = _features(df)
    platforms = sorted(df["platform"].unique())
    per = {p: n // len(platforms) + (1 if i < n % len(platforms) else 0) for i, p in enumerate(platforms)}
    chosen = []
    for p in platforms:
        idx = list(df.index[df["platform"] == p])
        seen: Counter = Counter()
        for _ in range(min(per[p], len(idx))):
            jitter = rng.random(len(idx)) * 0.5
            scores = []
            for pos, i in enumerate(idx):
                row = feats.loc[i]
                scores.append(sum(seen[(c, row[c])] for c in ("subtopic", "transcript", "duration", "lang")) + jitter[pos])
            best = idx.pop(int(np.argmin(scores)))
            for c in ("subtopic", "transcript", "duration", "lang"):
                seen[(c, feats.loc[best, c])] += 1
            chosen.append(best)
    out = df.loc[chosen].copy()
    out["stratum"] = out["platform"] + "|" + out["primary_subtopic"]
    out["stratum_size"] = np.nan
    out["sample_weight"] = np.nan
    return out
