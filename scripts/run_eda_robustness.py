#!/usr/bin/env python
"""Robustness checks for the EDA findings.

    python scripts/run_eda_robustness.py [--cap 3] [--seed 42]

1. Creator cap: keep at most N videos per creator (random) and repeat the key statistics, to see whether a few prolific creators drive the results.
2. Strict set only: human-labeled relevant videos vs model-selected relevant videos (do the two sources look alike?).

Writes reports/eda/robustness/robustness_report.md and tables/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.config import load_settings  # noqa: E402
from vidscreen.eda.features import build_features  # noqa: E402
from vidscreen.eda.runner import load_dataset  # noqa: E402
from vidscreen.eda.stats import compare_two, fmt_p  # noqa: E402


def cap_creators(d: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = d.assign(_r=rng.random(len(d)))
    out = d.sort_values("_r").groupby(["platform", "creator_id"], sort=False).head(cap).drop(columns="_r")
    return out.sort_index()


def key_stats(d: pd.DataFrame, label: str) -> dict:
    r = {"version": label, "videos": len(d), "youtube_share_pct": round((d.platform == "youtube").mean() * 100, 1), "creators": int(d.groupby(["platform", "creator_id"]).ngroups),
         "median_duration_s": float(d["duration_seconds"].median()), "median_views": float(d["views"].median()), "median_like_rate_pct": round(float(d["like_rate"].median()), 2),
         "with_transcript_pct": round(d["has_transcript"].mean() * 100, 1), "non_english_text_pct": round(d["non_english_text"].mean() * 100, 1)}
    if "screening_label" in d and d["screening_label"].nunique() > 1:
        r["relevant_pct"] = round((d["screening_label"] == "relevant").mean() * 100, 1)
    top = d["subtopic"].value_counts(normalize=True).head(3)
    r["top3_settings"] = "; ".join(f"{k} {v * 100:.0f}%" for k, v in top.items())
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    s = load_settings()
    out = s.root / "reports" / "eda" / "robustness"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    L = ["# EDA robustness checks", "", f"Creator cap: at most {a.cap} videos per creator and platform, chosen at random (seed {a.seed}). This is a sensitivity analysis, not a correction of the data.", ""]
    tables = []
    for ds in ("strict", "all"):
        raw = load_dataset(s, ds)
        if raw is None:
            continue
        d = build_features(raw)
        c = cap_creators(d, a.cap, a.seed)
        t = pd.DataFrame([key_stats(d, "all videos"), key_stats(c, f"cap {a.cap} per creator")])
        name = f"01_creator_cap_{ds}.csv"
        t.to_csv(out / "tables" / name, index=False, encoding="utf-8-sig"); tables.append(name)
        L += [f"## Creator cap: {ds} dataset", "", f"Kept {len(c):,} of {len(d):,} videos ({len(c) / len(d) * 100:.1f}%).", ""]
        for _, r in t.iterrows():
            L.append(f"- {r['version']}: " + f"{r['videos']:,} videos, YouTube {r['youtube_share_pct']}%, median duration {r['median_duration_s']:.0f} s, median views {r['median_views']:,.0f}, "
                     f"median like rate {r['median_like_rate_pct']}%, transcript {r['with_transcript_pct']}%" + (f", relevant {r['relevant_pct']}%" if "relevant_pct" in r and pd.notna(r["relevant_pct"]) else "") + f"; top settings: {r['top3_settings']}.")
        full = compare_two(d.loc[d.platform == "youtube", "duration_seconds"], d.loc[d.platform == "tiktok", "duration_seconds"], n_boot=500)
        capd = compare_two(c.loc[c.platform == "youtube", "duration_seconds"], c.loc[c.platform == "tiktok", "duration_seconds"], n_boot=500)
        if "cliffs_delta" in full and "cliffs_delta" in capd:
            L.append(f"- YouTube vs TikTok duration, Cliff's delta: {full['cliffs_delta']:+.2f} (all videos) vs {capd['cliffs_delta']:+.2f} (capped); {fmt_p(full['p_value'])} vs {fmt_p(capd['p_value'])}.")
        L.append("")
        if ds == "strict":
            h = d[d["label_source"] == "human"]; m = d[d["label_source"] == "model"]
            rows = [{"source": "human-labeled", **{k: v for k, v in key_stats(h, "").items() if k not in ("version", "top3_settings")}}, {"source": "model-selected", **{k: v for k, v in key_stats(m, "").items() if k not in ("version", "top3_settings")}}]
            ht = pd.DataFrame(rows)
            ht.to_csv(out / "tables" / "02_human_vs_model_strict.csv", index=False, encoding="utf-8-sig"); tables.append("02_human_vs_model_strict.csv")
            L += ["## Strict set: human-labeled vs model-selected", ""]
            for r in rows:
                L.append(f"- {r['source']}: {r['videos']} videos, median duration {r['median_duration_s']:.0f} s, median views {r['median_views']:,.0f}, transcript {r['with_transcript_pct']}%, non-English text {r['non_english_text_pct']}%.")
            ds_ = compare_two(h["duration_seconds"], m["duration_seconds"], n_boot=500)
            if "p_value" in ds_:
                L.append(f"- Duration (human vs model): median {ds_['median_x']:.0f} vs {ds_['median_y']:.0f} s, Cliff's delta {ds_['cliffs_delta']:+.2f} ({ds_['magnitude']}), {fmt_p(ds_['p_value'])}.")
            L += ["", "If the two sources differ a lot (for example in transcript share), the model-selected part of the strict set is a different kind of video from the human-labeled part, and conclusions about the strict set should say so.", ""]
    L += ["## Reading these results", "", "If platform shares, medians and top settings stay close after the creator cap, the main EDA findings do not depend on a few prolific creators. Search-result sampling can still favour certain creators, platforms and kinds of video.", "",
          "## Tables", ""] + [f"- `tables/{t}`" for t in tables]
    (out / "robustness_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out / "robustness_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
