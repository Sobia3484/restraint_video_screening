#!/usr/bin/env python
"""Draw a random comparison sample from the screening output groups.

  python scripts/make_comparison_sample.py --n 15 --platform tiktok
      reads  data/final/screening/irrelevant_videos.csv and audit_videos.csv
      writes data/final/checklist/comparison_sample.csv   (column "group" marks the source)

Then build the checklist workbook for the sample:
  python scripts/build_expert_checklist.py data/final/checklist/comparison_sample.csv data/final/checklist/Restraint_Checklist_Comparison_Sample.xlsx
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening-dir", default=str(ROOT / "data/final/screening"))
    ap.add_argument("--out", default=str(ROOT / "data/final/checklist/comparison_sample.csv"))
    ap.add_argument("--platform", default="tiktok", choices=["tiktok", "youtube", "both"])
    ap.add_argument("--n", type=int, default=15, help="videos per group")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    src = Path(a.screening_dir)
    parts = []
    for fname, group in (("irrelevant_videos.csv", "Irrelevant (screening)"), ("audit_videos.csv", "Audit (screening)")):
        d = pd.read_csv(src / fname)
        if a.platform != "both":
            d = d[d["platform"] == a.platform]
        if len(d) < a.n:
            sys.exit(f"{fname}: only {len(d)} videos for platform={a.platform}, need {a.n}")
        parts.append(d.sample(a.n, random_state=a.seed).assign(group=group))
    out = pd.concat(parts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"saved {a.out}: {len(out)} videos ({a.n} per group, platform={a.platform}, seed={a.seed})")


if __name__ == "__main__":
    main()