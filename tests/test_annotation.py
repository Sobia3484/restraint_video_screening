import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from vidscreen.annotation import (allocate, assign_primary_subtopic, build_adjudication_list, cohen_kappa,
                                  compare_annotators, diverse_sample, stratified_sample)
from vidscreen.annotation.workbook import SHOWN_COLUMNS, read_annotations, write_annotator_workbook


def make_master(n=200):
    rng = np.random.default_rng(0)
    subs = ["dental", "imaging", "general", "wound_care"]
    rows = []
    for plat in ("youtube", "tiktok"):
        for i in range(n):
            s = subs[rng.integers(4)]
            rows.append(dict(video_uid=f"{plat}:{i}", platform=plat, video_id=str(i), url="u", title="=1+1" if i == 0 else "t",
                             description="d", hashtags="a|b", transcript="x" * 20000 if i == 1 else "", creator_name="c",
                             duration_seconds="90", upload_date="2024-01-01", language_clean="en", has_transcript="False",
                             query_subtopics=f"{s}|general", query_ids="Q001", views="9"))
    return pd.DataFrame(rows)


class TestAnnotation(unittest.TestCase):
    def test_allocate_sums_and_caps(self):
        a = allocate({"a": 100, "b": 5, "c": 50}, 60, min_each=8)
        self.assertEqual(sum(a.values()), 60)
        self.assertLessEqual(a["b"], 5)
        self.assertEqual(allocate({"a": 3, "b": 4}, 100), {"a": 3, "b": 4})

    def test_kappa(self):
        self.assertEqual(cohen_kappa(list("aabb"), list("aabb")), 1.0)
        self.assertAlmostEqual(cohen_kappa(list("aabb"), list("abab")), 0.0)

    def test_primary_subtopic_uses_best_rank(self):
        m = make_master(2)
        qm = pd.DataFrame([
            dict(platform="youtube", video_id="0", query_rank="9", subtopic="dental"),
            dict(platform="youtube", video_id="0", query_rank="2", subtopic="imaging"),
        ])
        s = assign_primary_subtopic(m, qm)
        self.assertEqual(s[m.index[(m.platform == "youtube") & (m.video_id == "0")][0]], "imaging")

    def test_samples(self):
        m = make_master()
        m["primary_subtopic"] = assign_primary_subtopic(m, None)
        st = stratified_sample(m, 50, seed=1, min_each=3)
        self.assertEqual(st.groupby("platform").size().to_dict(), {"tiktok": 50, "youtube": 50})
        self.assertTrue((st["sample_weight"] > 0).all())
        pi = diverse_sample(m, 30, seed=1)
        self.assertEqual(len(pi), 30)
        self.assertEqual(pi["video_uid"].nunique(), 30)

    def test_workbook_hides_search_metadata_and_is_safe(self):
        m = make_master(5)
        m["primary_subtopic"] = "x"
        m.insert(0, "item_id", [f"P{i}" for i in range(len(m))])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.xlsx"
            write_annotator_workbook(m, p, 1)
            ws = load_workbook(p)["Annotation"]
            header = [c.value for c in ws[1]]
            for hidden in ("query_ids", "query_subtopics", "views", "primary_subtopic"):
                self.assertNotIn(hidden, header)
            self.assertIn("hashtags", header)
            self.assertTrue(all(h in header for h in SHOWN_COLUMNS))
            vals = [c.value for row in ws.iter_rows(min_row=2) for c in row]
            self.assertIn("=1+1", vals)                      # stored as text, not a formula
            self.assertTrue(all(len(str(v)) < 9000 for v in vals))  # long transcript truncated
            self.assertEqual(len(read_annotations(p)), len(m))

    def test_adjudication_list(self):
        key = pd.DataFrame(dict(item_id=["1", "2", "3"], platform="youtube", url="u", title="t"))
        cols = dict(age_ok="", topic_ok="", content_ok="", unsure_reason="", unsafe_demo="", note="")
        a = pd.DataFrame(dict(item_id=["1", "2", "3"], label=["Relevant", "Relevant", "Unsure"], **cols))
        b = pd.DataFrame(dict(item_id=["1", "2"], label=["Relevant", "Irrelevant"], **cols))
        adj = build_adjudication_list(key, a, b)
        self.assertEqual(sorted(adj["item_id"]), ["2", "3"])
        rep = compare_annotators(a[a.item_id.isin(b.item_id)], b)
        self.assertEqual(rep["percent_agreement"], 50.0)


if __name__ == "__main__":
    unittest.main()
