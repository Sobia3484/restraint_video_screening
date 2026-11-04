"""Merge data/raw/Q*.csv (complete or partial) into one master_raw.csv."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def read_csv_str(path: Path) -> pd.DataFrame:
    """Read a CSV with every column as text and blanks as '' (no silent type guessing)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _pick_files(raw_dir: Path) -> dict:
    chosen: dict = {}
    for p in sorted(Path(raw_dir).glob("Q*.csv")):
        qid = p.name.split("_", 1)[0]
        partial = p.name.endswith(".partial.csv")
        if qid not in chosen or (chosen[qid][1] and not partial):  # prefer the complete file
            chosen[qid] = (p, partial)
    return chosen


def merge_raw(raw_dir: Path, out_path: Path, expected_query_ids: Optional[Sequence[str]] = None):
    chosen = _pick_files(raw_dir)
    if not chosen:
        raise FileNotFoundError(f"No Q*.csv files found in {raw_dir}")
    frames = []
    for qid, (path, _partial) in sorted(chosen.items()):
        df = read_csv_str(path)
        df["source_file"] = path.name
        frames.append(df)
    master = pd.concat(frames, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False, encoding="utf-8-sig")
    missing = sorted(set(expected_query_ids or []) - set(chosen))
    report = {
        "files_merged": len(chosen),
        "partial_files": sorted(p.name for p, partial in chosen.values() if partial),
        "missing_queries": missing,
        "rows": int(len(master)),
        "rows_by_platform": master["platform"].value_counts().to_dict(),
        "output": str(out_path),
    }
    return master, report
