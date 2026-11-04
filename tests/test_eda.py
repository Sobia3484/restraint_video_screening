import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vidscreen.eda.features import build_features, detect_language
from vidscreen.eda.sections import SECTIONS
from vidscreen.eda.stats import cliffs_delta, compare_two, describe, gini, magnitude


def frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    titles = ["Papoose board in the dental chair", "Swaddling a 3 month old baby for a heel prick", "Pigg-O-Stat for infant chest X-ray", "Spinal motion restriction with a cervical collar for a child",
              "Comfort hold for toddler vaccination", "Distraction and child life for needle procedures"]
    rows = []
    for i in range(n):
        plat = "youtube" if i % 2 else "tiktok"
        views = int(10 ** rng.uniform(1, 6))
        lab = ["relevant", "audit", "irrelevant"][i % 3]
        rows.append({
            "video_uid": f"{plat}:v{i}", "platform": plat, "video_id": f"v{i}", "url": f"https://x/{i}", "title": titles[i % len(titles)] + f" {i}",
            "description": "A longer description about the topic" if plat == "youtube" else "", "creator_name": f"c{i % 30}", "creator_id": f"c{i % 30}",
            "creator_followers": float(10 ** rng.uniform(2, 6)), "duration_seconds": float(rng.integers(10, 900) if plat == "youtube" else rng.integers(5, 120)),
            "upload_date": f"202{rng.integers(0, 6)}-0{rng.integers(1, 9)}-1{rng.integers(0, 9)}", "views": views, "likes": int(views * rng.uniform(0.005, 0.05)),
            "comments": int(views * rng.uniform(0, 0.002)), "shares": np.nan if plat == "youtube" else int(views * 0.002), "saves": np.nan if plat == "youtube" else int(views * 0.004),
            "hashtags": "pediatrics|hospital" if i % 3 else "", "tags": "", "language_clean": "en", "country_clean": "Unknown", "query_count": int(rng.integers(1, 5)),
            "query_subtopics": "imaging|dental", "transcript": "", "transcript_status": "ok" if i % 5 == 0 else "no_caption_available",
            "transcript_word_count": 50 if i % 5 == 0 else 0, "label_source": "model" if i % 4 else "human", "zone": "relevant_confident" if lab == "relevant" else lab,
            "screening_label": lab, "score": float(rng.uniform(0, 1)), "retrieved_at_utc": "2026-10-31T17:00:00Z"})
    return pd.DataFrame(rows)


class StatsTests(unittest.TestCase):
    def test_cliffs_delta(self):
        self.assertAlmostEqual(cliffs_delta([3, 4, 5], [1, 2, 3]), 8 / 9, places=6)
        self.assertEqual(cliffs_delta([1, 2], [1, 2]), 0.0)

    def test_magnitude_and_gini(self):
        self.assertEqual(magnitude(0.1), "negligible")
        self.assertEqual(magnitude(-0.6), "large")
        self.assertAlmostEqual(gini([1, 1, 1, 1]), 0.0)

    def test_compare_and_describe(self):
        c = compare_two(np.arange(50), np.arange(50) + 30, n_boot=100)
        self.assertLess(c["p_value"], 0.01)
        self.assertEqual(describe([1, 2, 3])["median"], 2)


class FeatureTests(unittest.TestCase):
    def test_age_setting_and_techniques(self):
        d = build_features(frame())
        self.assertEqual(d[d.title.str.startswith("Swaddling")].age_group.iloc[0], "infant")
        self.assertEqual(d[d.title.str.startswith("Papoose")].subtopic.iloc[0], "dental")
        self.assertEqual(d[d.title.str.startswith("Pigg")].subtopic.iloc[0], "imaging (MRI, X-ray, CT)")
        self.assertEqual(d[d.title.str.startswith("Spinal")].subtopic.iloc[0], "trauma / spinal / splint")
        self.assertTrue(d[d.title.str.startswith("Swaddling")].tech_swaddling.all())
        self.assertTrue(d[d.title.str.startswith("Papoose")].tech_papoose_device.all())

    def test_older_child_and_transcript_flags(self):
        d = build_features(pd.concat([frame(6), frame(6).assign(title="Holding a 9 year old still for an X-ray", video_uid=lambda x: "z" + x.video_uid)]))
        self.assertEqual(d[d.title.str.startswith("Holding a 9")].age_group.iloc[0], "older child / adult mentioned")
        self.assertTrue(d["has_transcript"].any() and (~d["has_transcript"]).any())

    def test_language(self):
        self.assertEqual(detect_language("short")[0], "und")
        self.assertEqual(detect_language("Cómo inmovilizar a un bebé durante la radiografía para que no se mueva el niño")[1] in ("heuristic", "langdetect"), True)


class SectionTests(unittest.TestCase):
    def test_every_section_runs(self):
        d = build_features(frame())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name, fn in SECTIONS.items():
                sec = fn(d, out)
                self.assertTrue(sec.title, name)
            self.assertTrue((out / "tables").exists() and (out / "figures").exists())

    def test_outcome_needs_groups(self):
        d = build_features(frame())
        with tempfile.TemporaryDirectory() as tmp:
            one = SECTIONS["outcome"](d[d.screening_label == "relevant"], Path(tmp))
            self.assertTrue(one.notes and not one.findings)
            many = SECTIONS["outcome"](d, Path(tmp))
            self.assertTrue(many.findings)


if __name__ == "__main__":
    unittest.main()
