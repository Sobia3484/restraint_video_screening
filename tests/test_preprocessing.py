import tempfile
import unittest
from pathlib import Path

import pandas as pd

from vidscreen.csv_io import write_records
from vidscreen.preprocessing import clean_master, deduplicate, merge_raw
from vidscreen.preprocessing.clean import clean_country, clean_language, clean_text, repair_transcript
from vidscreen.schema import COLUMNS, VideoRecord


def rec(qid, vid, platform="youtube", **kw):
    base = dict(query_id=qid, query_text=qid, platform=platform, video_id=vid, query_rank=1, metadata_status="ok",
                url=f"https://www.youtube.com/watch?v={vid}", title="Infant CPR", duration_seconds=100,
                upload_date="2024-01-01", views=10, transcript_status="skipped_blocked")
    base.update(kw)
    return VideoRecord(**base)


class MergeDedupeTests(unittest.TestCase):
    def test_merge_prefers_complete_and_dedupe_keeps_best_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            write_records(raw / "Q001_a.partial.csv", [rec("Q001", "AAAAAAAAAAA"), rec("Q001", "BBBBBBBBBBB", query_rank=2)])
            write_records(raw / "Q001_a.csv", [rec("Q001", "AAAAAAAAAAA"), rec("Q001", "BBBBBBBBBBB", query_rank=2)])
            write_records(raw / "Q002_b.partial.csv", [rec("Q002", "AAAAAAAAAAA", transcript_status="ok", transcript="hi there", query_rank=5)])
            master, rep = merge_raw(raw, Path(tmp) / "m.csv", ["Q001", "Q002", "Q003"])
            self.assertEqual((rep["files_merged"], rep["rows"], rep["missing_queries"]), (2, 3, ["Q003"]))
            dedup, drep = deduplicate(master, {"Q001": "cpr", "Q002": "bls"})
            self.assertEqual(drep["unique_videos"], 2)
            a = dedup[dedup.video_id == "AAAAAAAAAAA"].iloc[0]
            self.assertEqual(a.transcript_status, "ok")           # the row with a transcript wins
            self.assertEqual(a.query_ids, "Q001|Q002")            # provenance preserved
            self.assertEqual((a.query_count, a.best_query_rank, a.query_subtopics), (2, 1, "bls|cpr"))


class CleanTests(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(clean_language("en-US"), "en")
        self.assertEqual(clean_language("un"), "unknown")
        self.assertEqual(clean_language(""), "unknown")
        self.assertEqual(clean_country("us"), "US")
        self.assertEqual(clean_country(""), "Unknown")
        self.assertEqual(clean_text("  a\x00b \r\n c "), "ab \n c")

    def test_repair_transcript(self):
        garbled = "doctora mi beb\u00c3\u00a9 se ahoga"           # UTF-8 bytes read as Latin-1
        self.assertEqual(repair_transcript(garbled), ("doctora mi beb\u00e9 se ahoga", "repaired"))
        self.assertEqual(repair_transcript("plain ascii"), ("plain ascii", "unchanged"))
        self.assertEqual(repair_transcript("\u06cc\u06c1 \u0628\u0686\u06c1"), ("\u06cc\u06c1 \u0628\u0686\u06c1", "unchanged"))
        self.assertEqual(repair_transcript("\u00db\x8c\u00db \u00d8\u00ba")[1], "corrupt")  # lost bytes -> unrecoverable

    def test_drop_rules_and_typing(self):
        rows = [
            rec("Q001", "AAAAAAAAAAA", language="en-US", views=-5),
            rec("Q001", "BBBBBBBBBBB", metadata_status="unavailable: Private video", title=""),
            rec("Q001", "CCCCCCCCCCC", duration_seconds=None),
            rec("Q001", "DDDDDDDDDDD", title="", description="", transcript=""),
            rec("Q001", "EEEEEEEEEEE", upload_date="2099-01-01"),
            rec("Q001", "t1", platform="tiktok", url="https://www.tiktok.com/@u/video/123", country_code="us"),
        ]
        df = pd.DataFrame([r.to_row() for r in rows], columns=COLUMNS).astype(str)
        df = df.replace({"None": ""})
        df["query_ids"], df["query_count"], df["best_query_rank"], df["query_subtopics"] = "Q001", "1", "1", "cpr"
        cleaned, dropped, rep = clean_master(df)
        self.assertEqual(set(dropped.reason), {"metadata_not_ok", "invalid_duration", "no_text_at_all"})
        self.assertEqual(len(cleaned), 3)
        a = cleaned[cleaned.video_id == "AAAAAAAAAAA"].iloc[0]
        self.assertTrue(pd.isna(a.views))                       # negative -> missing
        self.assertEqual(a.language_clean, "en")
        e = cleaned[cleaned.video_id == "EEEEEEEEEEE"].iloc[0]
        self.assertEqual(e.upload_date, "")                     # future date blanked, row kept
        t = cleaned[cleaned.platform == "tiktok"].iloc[0]
        self.assertEqual((t.country_clean, t.video_uid), ("US", "tiktok:t1"))
        self.assertTrue(pd.isna(t.has_description))              # not applicable to TikTok
        self.assertEqual(a.transcript_language_clean, "")        # no transcript -> blank, not "unknown"


if __name__ == "__main__":
    unittest.main()
