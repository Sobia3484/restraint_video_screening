"""One row per unique video; every query that found it is preserved."""
from __future__ import annotations

from typing import Optional

import pandas as pd

KEY = ["platform", "video_id"]
_BIG = 10**6


def build_query_map(master: pd.DataFrame, query_meta: Optional[dict] = None) -> pd.DataFrame:
    """Long table: which query returned which video at which rank (for query-level EDA)."""
    cols = ["query_id", "query_text", "platform", "video_id", "query_rank"]
    out = master[cols].copy()
    out["subtopic"] = out["query_id"].map(query_meta or {}).fillna("")
    return out.drop_duplicates(["query_id", "platform", "video_id"]).reset_index(drop=True)


def deduplicate(master: pd.DataFrame, query_meta: Optional[dict] = None):
    """Collapse duplicates on (platform, video_id).

    The kept row is the most complete one (metadata ok > transcript ok > best rank).
    Query provenance is aggregated into query_ids / query_count / best_query_rank / query_subtopics.
    """
    df = master.copy()
    blank = df["video_id"].str.strip() == ""
    n_blank = int(blank.sum())
    df = df[~blank].copy()

    df["_meta_ok"] = df["metadata_status"] == "ok"
    df["_tr_ok"] = df["transcript_status"] == "ok"
    df["_rank"] = pd.to_numeric(df["query_rank"], errors="coerce").fillna(_BIG)

    best = (
        df.sort_values(["_meta_ok", "_tr_ok", "_rank"], ascending=[False, False, True], kind="stable")
        .drop_duplicates(KEY, keep="first")
    )
    agg = (
        df.groupby(KEY, sort=False)
        .agg(
            query_ids=("query_id", lambda s: "|".join(sorted(set(s)))),
            query_count=("query_id", "nunique"),
            best_query_rank=("_rank", "min"),
        )
        .reset_index()
    )
    agg["best_query_rank"] = agg["best_query_rank"].where(agg["best_query_rank"] < _BIG).astype("Int64")
    meta = query_meta or {}
    agg["query_subtopics"] = agg["query_ids"].map(
        lambda ids: "|".join(sorted({meta.get(q, "") for q in ids.split("|")} - {""}))
    )
    drop_cols = ["query_id", "query_text", "query_rank", "source_file", "_meta_ok", "_tr_ok", "_rank"]
    dedup = best.drop(columns=[c for c in drop_cols if c in best.columns]).merge(agg, on=KEY, how="left")
    dedup = dedup.sort_values(KEY, kind="stable").reset_index(drop=True)

    per_platform = {}
    for platform, g in dedup.groupby("platform"):
        rows_in = int((df["platform"] == platform).sum())
        per_platform[platform] = {
            "rows_in": rows_in,
            "unique_videos": int(len(g)),
            "duplicates_removed": rows_in - int(len(g)),
            "videos_found_by_2plus_queries": int((g["query_count"] >= 2).sum()),
            "videos_found_by_5plus_queries": int((g["query_count"] >= 5).sum()),
            "max_queries_for_one_video": int(g["query_count"].max()),
            "mean_queries_per_video": round(float(g["query_count"].mean()), 2),
        }
    report = {
        "rows_in": int(len(master)),
        "rows_dropped_missing_video_id": n_blank,
        "unique_videos": int(len(dedup)),
        "duplicates_removed": int(len(master) - n_blank - len(dedup)),
        "by_platform": per_platform,
    }
    return dedup, report
