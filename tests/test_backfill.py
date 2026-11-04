import csv
import tempfile
import unittest
from pathlib import Path

from vidscreen.backfill import backfill_csv, backfill_dir, count_missing, run_with_cooldown
from vidscreen.csv_io import write_records
from vidscreen.schema import VideoRecord
from vidscreen.transcripts import TranscriptResult


def rec(vid, status, platform="youtube", meta="ok"):
    return VideoRecord(query_id="Q001", query_text="x", platform=platform, video_id=vid,
                       metadata_status=meta, transcript_status=status)


class FakeFetcher:
    def __init__(self, block_after=None):
        self.halted, self.calls, self.block_after = False, [], block_after

    def fetch(self, vid):
        if self.block_after is not None and len(self.calls) >= self.block_after:
            self.halted = True
            return TranscriptResult(status="blocked")
        self.calls.append(vid)
        return TranscriptResult(text="hello world", status="ok", source="youtube_captions", language="en", is_auto=True)


class RoundFetcher:
    """Allows `per_round` fetches, then reports a block until reset_halt() (a cooldown) is called."""

    def __init__(self, per_round):
        self.per_round, self.used, self.halted = per_round, 0, False

    def fetch(self, vid):
        if self.used >= self.per_round:
            self.halted = True
            return TranscriptResult(status="blocked")
        self.used += 1
        return TranscriptResult(text="hi", status="ok", source="youtube_captions", language="en", is_auto=False)

    def reset_halt(self):
        self.used, self.halted = 0, False


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class BackfillTests(unittest.TestCase):
    def test_fills_only_retryable_rows_and_renames(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "Q001_x.partial.csv"
            write_records(p, [rec("a", "skipped_blocked"), rec("b", "ok"), rec("c", "no_transcript"),
                              rec("d", "blocked"), rec("t1", "no_caption_available", platform="tiktok"),
                              rec("u", "skipped_video_unavailable", meta="unavailable: x")])
            f = FakeFetcher()
            stats = backfill_csv(p, f)
            self.assertEqual(f.calls, ["a", "d"])
            self.assertEqual((stats["updated"], stats["remaining"]), (2, 0))
            self.assertEqual(stats["path"].name, "Q001_x.csv")
            self.assertFalse(p.exists())
            rows = {r["video_id"]: r for r in read(stats["path"])}
            self.assertEqual(rows["a"]["transcript"], "hello world")
            self.assertEqual(rows["a"]["transcript_word_count"], "2")
            self.assertEqual(rows["a"]["transcript_is_auto"], "True")
            self.assertEqual(rows["c"]["transcript_status"], "no_transcript")  # untouched

    def test_stops_when_blocked_and_stays_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "Q001_x.partial.csv"
            write_records(p, [rec(str(i), "skipped_blocked") for i in range(4)])
            f = FakeFetcher(block_after=2)
            stats = backfill_csv(p, f)
            self.assertEqual((stats["updated"], stats["remaining"]), (2, 2))
            self.assertTrue(p.exists())
            results = backfill_dir(Path(tmp), f)  # halted fetcher -> nothing more happens
            self.assertEqual(results, [])


class CooldownTests(unittest.TestCase):
    def _make(self, tmp, n):
        write_records(Path(tmp) / "Q001_x.partial.csv", [rec(f"v{i}", "skipped_blocked") for i in range(n)])

    def test_cooldown_rounds_finish_the_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make(tmp, 5)
            self.assertEqual(count_missing(Path(tmp)), (5, 5))
            sleeps = []
            out = run_with_cooldown(Path(tmp), RoundFetcher(2), cooldown_seconds=60, sleep_fn=sleeps.append)
            self.assertEqual((out["status"], out["rounds"], out["filled_rows"], out["missing_rows"]), ("complete", 3, 5, 0))
            self.assertEqual(sleeps, [60, 60])
            self.assertTrue((Path(tmp) / "Q001_x.csv").exists())

    def test_stops_when_cooldown_does_not_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make(tmp, 3)
            out = run_with_cooldown(Path(tmp), RoundFetcher(0), cooldown_seconds=60, stall_limit=2, sleep_fn=lambda s: None)
            self.assertEqual((out["status"], out["rounds"], out["missing_rows"]), ("stalled", 2, 3))

    def test_without_cooldown_single_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make(tmp, 4)
            out = run_with_cooldown(Path(tmp), RoundFetcher(1), cooldown_seconds=0)
            self.assertEqual((out["status"], out["rounds"], out["filled_rows"], out["missing_rows"]), ("stopped_blocked", 1, 1, 3))


if __name__ == "__main__":
    unittest.main()
