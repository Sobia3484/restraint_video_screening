"""Numeric / signal features (no views, likes, query information or transcript).

Duration is z-scored within each platform. Everything is computed without looking at labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import ADULT, CHILD, INSTRUCT, PROMO, SETTING, SLEEP, TOPIC, exclusion_category
from .text import fold

NUMERIC = [
    "duration_z", "title_len_log", "desc_len_log", "has_description", "n_hashtags", "n_tags",
    "child_n", "topic_n", "topic_hit", "setting_hit", "instruct_hit", "promo_hit", "adult_hit", "sleep_hit", "excl_hit", "is_tiktok",
    "has_transcript", "transcript_len_log",
]


def _count(rx, texts) -> np.ndarray:
    return np.array([len({m.group(0) for m in rx.finditer(t)}) for t in texts], dtype=float)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    allt = [fold(f"{t} {h.replace('|', ' ')} {g.replace('|', ' ')} {x or ''}")
            for t, h, g, x in zip(d["title"], d["hashtags"], d["tags"], d["description"])]
    dur = pd.to_numeric(d["duration_seconds"], errors="coerce").fillna(0)
    log_dur = np.log1p(dur)
    grp = log_dur.groupby(d["platform"])
    d["duration_z"] = ((log_dur - grp.transform("mean")) / grp.transform("std").replace(0, 1)).fillna(0)
    d["title_len_log"] = np.log1p(d["title"].str.len())
    d["desc_len_log"] = np.log1p(d["description"].str.len())
    d["has_description"] = (d["description"].str.len() > 0).astype(float)
    d["n_hashtags"] = d["hashtags"].map(lambda s: len([x for x in s.split("|") if x]))
    d["n_tags"] = d["tags"].map(lambda s: len([x for x in s.split("|") if x]))
    d["child_n"] = _count(CHILD, allt)
    d["topic_n"] = _count(TOPIC, allt)
    d["topic_hit"] = (d["topic_n"] > 0).astype(float)
    for name, rx in (("setting_hit", SETTING), ("instruct_hit", INSTRUCT), ("promo_hit", PROMO), ("adult_hit", ADULT), ("sleep_hit", SLEEP)):
        d[name] = (_count(rx, allt) > 0).astype(float)
    d["excl_hit"] = np.array([1.0 if exclusion_category(t) else 0.0 for t in allt])
    tr = d["transcript"] if "transcript" in d else pd.Series("", index=d.index)
    d["has_transcript"] = (tr.str.len() > 0).astype(float)
    d["transcript_len_log"] = np.log1p(tr.str.len())
    d["is_tiktok"] = (d["platform"] == "tiktok").astype(float)
    d["word_count"] = d["text"].str.split().str.len()
    return d