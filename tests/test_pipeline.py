import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from vidscreen.config import load_settings
from vidscreen.pipeline import Pipeline
from vidscreen.queries import Query
from vidscreen.schema import COLUMNS
from vidscreen.transcripts import TranscriptResult

csv.field_size_limit(2**31 - 1)  # sys.maxsize overflows on Windows
ROOT = Path(__file__).resolve().parents[1]


class FakeYouTube:
    def search(self, q):
        return ["vid1", "vid2", "vid3"]

    def fetch_video(self, vid):
        if vid == "vid3":
            return {"id": vid, "_status": "unavailable", "_error": "Private video"}
        return {"_status": "ok", "id": vid, "title": f"Title {vid}", "description": "d\nwith newline, and comma",
                "channel": "Ch", "channel_id": "UC", "duration": 100, "view_count": 10, "upload_date": "20240101"}


class FakeYTTranscripts:
    halted = False

    def fetch(self, vid):
        return TranscriptResult(text="hello उर्दू world", status="ok", source="youtube_captions", language="en", is_auto=False)


class FakeTikTok:
    def __init__(self, limit_hit=False):
        self.credits_spent = 0
        self.credits_remaining = None
        self.limit_hit = limit_hit

    def search(self, q):
        self.credits_spent += 1
        return [{"aweme_id": "t1", "desc": "caption #cpr", "author": {"unique_id": "u"}, "statistics": {"play_count": 5},
                 "video": {"duration": 12000}, "create_time": 1707724682}]


class FakeCaptions:
    def fetch(self, vid, aweme):
        return TranscriptResult(status="no_caption_available")


def make_pipeline(tmp, tiktok=None, **kw):
    s = replace(load_settings(root=ROOT), raw_dir=Path(tmp), cache_dir=Path(tmp) / "cache", yt_workers=2, tt_api_key="x")
    return Pipeline(s, youtube=FakeYouTube(), yt_transcripts=FakeYTTranscripts(),
                    tiktok=tiktok or FakeTikTok(), tt_captions=FakeCaptions(), **kw), s


Q1 = Query("Q001", "Pediatric CPR", "cpr")


class PipelineTests(unittest.TestCase):
    def test_one_csv_per_query_with_both_platforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe, s = make_pipeline(tmp)
            rows = pipe.run([Q1])
            self.assertEqual(rows[0]["status"], "complete")
            path = Path(tmp) / "Q001_pediatric_cpr.csv"
            with path.open(encoding="utf-8-sig", newline="") as f:
                data = list(csv.DictReader(f))
            self.assertEqual(list(data[0].keys()), COLUMNS)
            self.assertEqual(len(data), 4)  # 3 youtube + 1 tiktok
            self.assertEqual({r["platform"] for r in data}, {"youtube", "tiktok"})
            self.assertEqual(data[0]["transcript"], "hello उर्दू world")
            self.assertIn("newline", data[0]["description"])
            self.assertEqual(data[2]["metadata_status"][:11], "unavailable")
            self.assertEqual(data[3]["duration_seconds"], "12.0")
            self.assertTrue((Path(tmp) / "_manifest.csv").exists())

    def test_existing_csv_is_skipped_unless_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe, _ = make_pipeline(tmp)
            self.assertEqual(len(pipe.run([Q1])), 1)
            self.assertEqual(len(pipe.run([Q1])), 0)  # skipped
            pipe_force, _ = make_pipeline(tmp, force=True)
            self.assertEqual(len(pipe_force.run([Q1])), 1)

    def test_credit_limit_marks_partial_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe, _ = make_pipeline(tmp, tiktok=FakeTikTok(limit_hit=True))
            rows = pipe.run([Q1])
            self.assertEqual(rows[0]["status"], "partial")
            self.assertTrue((Path(tmp) / "Q001_pediatric_cpr.partial.csv").exists())
            self.assertFalse((Path(tmp) / "Q001_pediatric_cpr.csv").exists())
            pipe2, _ = make_pipeline(tmp)  # next run, credits available
            rows2 = pipe2.run([Q1])
            self.assertEqual(rows2[0]["status"], "complete")
            self.assertFalse((Path(tmp) / "Q001_pediatric_cpr.partial.csv").exists())

    def test_youtube_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = replace(load_settings(root=ROOT), raw_dir=Path(tmp), cache_dir=Path(tmp) / "c", yt_workers=1)
            pipe = Pipeline(s, platforms=("youtube",), youtube=FakeYouTube(), yt_transcripts=FakeYTTranscripts())
            rows = pipe.run([Q1])
            self.assertEqual((rows[0]["youtube_rows"], rows[0]["tiktok_rows"]), (3, 0))


if __name__ == "__main__":
    unittest.main()
