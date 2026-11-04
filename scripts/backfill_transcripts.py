#!/usr/bin/env python
"""Fill the YouTube transcripts that were blocked during extraction.

    python scripts/backfill_transcripts.py                          # one round, stops when blocked
    python scripts/backfill_transcripts.py --cooldown-minutes 20    # keeps going: waits, then retries

Safe to stop (Ctrl+C) and re-run at any time: finished transcripts are cached and every CSV is saved as it
is updated. No TikTok credits are used. Completed `.partial.csv` files are renamed to `.csv` automatically.
After it finishes, run:  python scripts/run_preprocessing.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging  # noqa: E402

from vidscreen.backfill import count_missing, run_with_cooldown  # noqa: E402
from vidscreen.cache import JsonCache  # noqa: E402
from vidscreen.config import load_settings  # noqa: E402
from vidscreen.logging_setup import setup_logging  # noqa: E402
from vidscreen.transcripts import YouTubeTranscriptFetcher  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cooldown-minutes", type=float, default=0, help="After a block, wait this long and retry (0 = stop)")
    ap.add_argument("--max-hours", type=float, default=10, help="Overall time limit when using --cooldown-minutes")
    ap.add_argument("--sleep", type=float, help="Seconds between requests (default from settings, 3)")
    args = ap.parse_args()

    s = load_settings()
    setup_logging(s.log_dir)
    log = logging.getLogger("vidscreen")
    rows, unique = count_missing(s.raw_dir)
    log.info("Missing YouTube transcripts: %d rows (%d unique videos)", rows, unique)
    fetcher = YouTubeTranscriptFetcher(
        s.tr_languages, JsonCache(s.cache_dir), sleep=args.sleep or s.tr_yt_sleep,
        block_threshold=s.tr_block_threshold, workers=1,
    )
    try:
        out = run_with_cooldown(s.raw_dir, fetcher, args.cooldown_minutes * 60, args.max_hours * 3600)
    except KeyboardInterrupt:
        rows, unique = count_missing(s.raw_dir)
        log.info("Stopped by you. Progress is saved. Still missing: %d rows (%d unique videos).", rows, unique)
        return 1
    log.info("Finished (%s): %d rounds, %d rows filled, still missing %d rows (%d unique videos).",
             out["status"], out["rounds"], out["filled_rows"], out["missing_rows"], out["missing_unique_videos"])
    if out["missing_rows"]:
        log.info("To continue: switch network / toggle airplane mode on your phone, then run this script again.")
    return 0 if not out["missing_rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
