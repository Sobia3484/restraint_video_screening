#!/usr/bin/env python
"""Tiny end-to-end check of every tool before the full run.

Runs ONE query (Q001) with 3 YouTube videos + 1 TikTok page (= 1 credit),
writes to data/smoke_test/, and prints what came back. The TikTok page is cached, so the
credit is not wasted: the full run reuses it.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
csv.field_size_limit(2**31 - 1)  # sys.maxsize overflows on Windows

from vidscreen.cli import main  # noqa: E402


def show(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"\n=== {path.name}: {len(rows)} rows ===")
    for platform in ("youtube", "tiktok"):
        subset = [r for r in rows if r["platform"] == platform]
        ok_tr = sum(r["transcript_status"] == "ok" for r in subset)
        print(f"{platform:8s} rows={len(subset)}  with_transcript={ok_tr}")
        for r in subset[:3]:
            print(f"   - {r['video_id']} | {r['duration_seconds']}s | views={r['views']} | "
                  f"likes={r['likes']} | date={r['upload_date']} | transcript={r['transcript_status']}")
            print(f"     {r['title'][:90]!r}")


if __name__ == "__main__":
    code = main(["--queries", "Q001", "--yt-results", "3", "--tt-pages", "1",
                 "--output-dir", "data/smoke_test", "--force"] + sys.argv[1:])
    for csv_path in sorted((ROOT / "data" / "smoke_test").glob("Q001_*.csv")):
        show(csv_path)
    print("\nCheck: durations look right (TikTok in seconds?), views/likes filled, transcripts 'ok' for some rows.")
    sys.exit(code)
