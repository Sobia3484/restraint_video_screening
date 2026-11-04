from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

from .schema import COLUMNS, VideoRecord

MANIFEST_COLUMNS = [
    "run_started_utc", "query_id", "query_text", "output_file", "status",
    "youtube_rows", "youtube_with_transcript", "tiktok_rows", "tiktok_with_transcript",
    "tiktok_credits_spent", "notes",
]


def write_records(path: Path, records: Iterable[VideoRecord]) -> int:
    """Atomically write records to CSV (UTF-8 with BOM so Excel shows Urdu/Hindi correctly)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_row())
            n += 1
    os.replace(tmp, path)
    return n


def append_manifest(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
