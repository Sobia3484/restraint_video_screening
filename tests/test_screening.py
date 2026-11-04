import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from vidscreen.config import load_settings
from vidscreen.screening.features import add_features
from vidscreen.screening.rules import apply_rules
from vidscreen.screening.run import (Spec, ablation_summary, apply_calibrator, base_specs, fit_calibrator, run,
                                     strict_threshold, zone_of)
from vidscreen.screening.text import build_text

ROOT = Path(__file__).resolve().parents[1]


def fake_encoder(texts):
    out = np.zeros((len(texts), 64))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, sum(map(ord, w)) % 64] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


REL = ["Papoose board for toddler dental treatment {}", "How to hold a baby still for a vaccine {}", "Swaddling an infant for MRI {}",
       "Cervical collar and spinal motion restriction in a child {}", "Clinical holding of a child for blood draw {}"]
IRR = ["Restraint collapse after school: why kids melt down {}", "Baby sleep swaddle tutorial {}", "Car seat installation guide {}",
       "Official music video lyrics {}", "Cooking pasta {}", "Adult nursing home care {}"]


def make_world(tmp: Path, n=300):
    rng = np.random.default_rng(1)
    rows, labels = [], []
    for i in range(n):
        is_rel = bool(rng.random() < 0.3)
        plat = "youtube" if i % 2 else "tiktok"
        uid = f"{plat}:v{i}"
        title = (REL if is_rel else IRR)[int(rng.integers(0, len(REL if is_rel else IRR)))].format(i)
        rows.append({"video_uid": uid, "platform": plat, "video_id": f"v{i}", "url": f"https://x/{i}", "creator_id": f"c{i % 60}", "title": title,
                     "description": "d" if plat == "youtube" else "", "hashtags": "fyp" if plat == "tiktok" else "", "tags": "", "transcript": ("how to hold the child still " if is_rel and i % 4 else "") if plat == "tiktok" else "",
                     "duration_seconds": str(int(rng.integers(20, 600))), "query_ids": "Q001", "query_subtopics": "general"})
        if i % 3 != 0:
            labels.append({"video_uid": uid, "label_final": "Relevant" if is_rel else "Irrelevant", "sample_weight": "2.0"})
    (tmp / "int").mkdir()
    (tmp / "final").mkdir()
    pd.DataFrame(rows).to_csv(tmp / "int" / "master_cleaned.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(labels).to_csv(tmp / "final" / "labeled_set.csv", index=False, encoding="utf-8-sig")
    return len(labels)


def settings_for(tmp: Path):
    return replace(load_settings(root=ROOT), root=tmp, intermediate_dir=tmp / "int", final_dir=tmp / "final", cache_dir=tmp / "cache")


class RuleTests(unittest.TestCase):
    def label(self, title, desc="", tags=""):
        return apply_rules(title, desc, "", tags)

    def test_relevant_and_gates(self):
        r = self.label("How to hold a baby still for a vaccine")
        self.assertEqual(r.label, "relevant")
        self.assertTrue(r.gate_strict)
        r = self.label("Papoose board in dentistry")           # topic, no child term
        self.assertTrue(r.gate_topic)
        self.assertFalse(r.gate_strict)
        self.assertEqual(r.label, "potentially_relevant")

    def test_false_matches(self):
        self.assertEqual(self.label("Restraint collapse after school: why kids fall apart").label, "irrelevant")
        self.assertEqual(self.label("Child restraint car seat installation").label, "irrelevant")
        self.assertEqual(self.label("Immobilize lyrics official music video").label, "irrelevant")
        self.assertEqual(self.label("Swaddle your baby for sleep tutorial").reason, "sleep_context")
        self.assertFalse(self.label("Swaddle your baby for sleep tutorial").gate_topic)
        self.assertTrue(self.label("Swaddling an infant for MRI scan").gate_strict)   # procedural swaddling stays
        self.assertTrue(self.label("Infant restraint for CAT scan").gate_strict)      # 'cat scan' is not a pet

    def test_veto_gate_only_blocks_false_matches(self):
        self.assertTrue(self.label("Funny toddler dances").blocked is False)
        self.assertTrue(self.label("Car seat for baby").blocked)

    def test_no_topic(self):
        self.assertEqual(self.label("Funny toddler dances").label, "irrelevant")


class StrictThresholdTests(unittest.TestCase):
    def test_threshold_meets_target_and_respects_floor(self):
        rng = np.random.default_rng(0)
        y = (rng.random(400) < 0.3).astype(int)
        p = np.clip(y * 0.55 + rng.random(400) * 0.45, 0, 1)
        t, ok = strict_threshold(y, p, np.ones(400), 0.9, 0.8)
        self.assertGreaterEqual(t, 0.8)
        if ok:
            self.assertGreaterEqual(float(y[p >= t].mean()), 0.9)

    def test_unreachable_target_is_flagged(self):
        y = np.array([0, 1] * 50)
        p = np.linspace(0, 1, 100)
        t, ok = strict_threshold(y, p, np.ones(100), 1.0, 0.8)
        self.assertFalse(ok)

    def test_zones(self):
        z = zone_of(np.array([0.05, 0.5, 0.9]), 0.8, 0.15)
        self.assertEqual(list(z), ["irrelevant_confident", "audit", "relevant_confident"])


class FeatureTests(unittest.TestCase):
    def test_duration_normalised_within_platform(self):
        df = pd.DataFrame({"platform": ["tiktok"] * 3 + ["youtube"] * 3, "duration_seconds": [10, 20, 40, 600, 1200, 2400],
                           "title": ["Infant restraint"] * 6, "description": [""] * 6, "hashtags": [""] * 6, "tags": [""] * 6})
        df["text"] = df.apply(build_text, axis=1)
        d = add_features(df)
        self.assertAlmostEqual(float(d[d.platform == "tiktok"].duration_z.mean()), 0.0, places=6)


class EndToEndTests(unittest.TestCase):
    def test_run_all(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            n_lab = make_world(tmp)
            s = settings_for(tmp)
            rep = run(s, "all", True, encoder=fake_encoder, repeats=1, folds=3)
            out = tmp / "final" / "screening"
            for f in ("screening_report.md", "model_comparison.csv", "screening_results.csv", "relevant_strict_videos.csv", "audit_videos.csv"):
                self.assertTrue((out / f).exists(), f)
            self.assertEqual(rep["n_labeled"], n_lab)
            self.assertEqual(len(pd.read_csv(out / "model_comparison.csv")), 48)
            res = pd.read_csv(out / "screening_results.csv")
            self.assertEqual(len(res), 300)
            model = res[res.label_source == "model"]
            high = rep["strict"]["high_threshold"]
            # strictness: a model-scored video is 'relevant' only at or above the strict threshold; audit is never relevant
            self.assertTrue((model[model.screening_label == "relevant"].score >= high).all())
            self.assertFalse(((model.zone == "audit") & (model.screening_label == "relevant")).any())
            strict = pd.read_csv(out / "relevant_strict_videos.csv")
            self.assertTrue(set(strict.screening_label) <= {"relevant"})

    def test_cv_stage_has_no_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            make_world(tmp)
            run(settings_for(tmp), "cv", False, encoder=None, repeats=1, folds=3)
            self.assertFalse((tmp / "final" / "screening" / "screening_results.csv").exists())

    def test_ablation_and_specs(self):
        self.assertEqual(len(base_specs(True)), 48)
        self.assertEqual(len(base_specs(False)), 24)


if __name__ == "__main__":
    unittest.main()