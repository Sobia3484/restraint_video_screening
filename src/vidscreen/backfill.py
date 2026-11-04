"""Fill missing YouTube transcripts in already-written CSVs (no re-search, no TikTok credits).

A row needs a retry when its transcript was blocked by YouTube (`blocked`, `skipped_blocked`)
or failed with an unexpected error (`error:*`). Definitive outcomes (ok, no_transcript,
disabled, ...) are never touched. When a `.partial.csv` has nothing left to retry it is
renamed to the final `.csv`.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

from .utils import word_count

log = logging.getLogger("vidscreen")
csv.field_size_limit(2**31 - 1)  # sys.maxsize overflows on Windows
RETRY_STATUSES = {"blocked", "skipped_blocked"}


def needs_retry(row: dict) -> bool:
    status = row.get("transcript_status", "")
    return (
        row.get("platform") == "youtube"
        and row.get("metadata_status") == "ok"
        and (status in RETRY_STATUSES or status.startswith("error:"))
    )


def _is_retry_status(status: str) -> bool:
    return status in RETRY_STATUSES or status.startswith("error:")


def backfill_csv(path: Path, fetcher) -> dict:
    """Update one CSV in place. Returns stats; `path` key is the (possibly renamed) file."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    updated = 0
    for row in rows:
        if not needs_retry(row):
            continue
        if fetcher.halted:
            break
        res = fetcher.fetch(row["video_id"])
        if _is_retry_status(res.status):
            continue  # still blocked/failing: leave the row as it was
        row["transcript"] = res.text or ""
        row["transcript_status"] = res.status
        row["transcript_source"] = res.source
        row["transcript_language"] = res.language
        row["transcript_is_auto"] = "" if res.is_auto is None else ("True" if res.is_auto else "False")
        row["transcript_word_count"] = word_count(res.text) if res.text else 0
        updated += 1

    remaining = sum(needs_retry(r) for r in rows)
    if updated:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)

    final_path = path
    if remaining == 0 and path.name.endswith(".partial.csv"):
        final_path = path.with_name(path.name.replace(".partial.csv", ".csv"))
        os.replace(path, final_path)
        log.info("%s is now complete -> %s", path.name, final_path.name)
    return {"path": final_path, "updated": updated, "remaining": remaining, "rows": len(rows)}


def backfill_dir(raw_dir: Path, fetcher) -> list[dict]:
    files = sorted(Path(raw_dir).glob("Q*.csv"))  # matches both .csv and .partial.csv
    results = []
    for path in files:
        if fetcher.halted:
            log.warning("YouTube is blocking again - stopping. Run this script again later; progress is saved.")
            break
        stats = backfill_csv(path, fetcher)
        log.info("%s: filled %d, still missing %d", stats["path"].name, stats["updated"], stats["remaining"])
        results.append(stats)
    return results


def count_missing(raw_dir: Path) -> tuple:
    """(rows, unique_videos) that still need a YouTube transcript, across ALL CSVs."""
    rows, ids = 0, set()
    for path in sorted(Path(raw_dir).glob("Q*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if needs_retry(row):
                    rows += 1
                    ids.add(row["video_id"])
    return rows, len(ids)


def run_with_cooldown(
    raw_dir: Path,
    fetcher,
    cooldown_seconds: float = 0,
    max_seconds: float = 36000,
    stall_limit: int = 3,
    sleep_fn=time.sleep,
    clock=time.monotonic,
) -> dict:
    """Repeat backfill rounds; when YouTube blocks, wait `cooldown_seconds` and try again.

    Stops when: everything is filled, cooldown is off, no progress for `stall_limit` rounds,
    or the time limit would be exceeded. Progress is saved after every file.
    """
    start, rounds, stalled, total = clock(), 0, 0, 0
    while True:
        rounds += 1
        results = backfill_dir(raw_dir, fetcher)
        filled = sum(r["updated"] for r in results)
        total += filled
        rows, unique = count_missing(raw_dir)
        log.info("Round %d: filled %d rows | still missing %d rows (%d unique videos)", rounds, filled, rows, unique)
        if rows == 0:
            status = "complete"
            break
        if not fetcher.halted and filled == 0:
            status = "no_progress"  # not blocked, yet nothing could be filled
            break
        if not cooldown_seconds:
            status = "stopped_blocked"
            break
        stalled = stalled + 1 if filled == 0 else 0
        if stalled >= stall_limit:
            status = "stalled"
            break
        if clock() - start + cooldown_seconds > max_seconds:
            status = "time_limit"
            break
        log.info("Cooling down for %.0f minutes before the next round (Ctrl+C to stop; progress is saved)...", cooldown_seconds / 60)
        sleep_fn(cooldown_seconds)
        fetcher.reset_halt()
    return {"status": status, "rounds": rounds, "filled_rows": total, "missing_rows": rows, "missing_unique_videos": unique}
