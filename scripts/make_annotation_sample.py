#!/usr/bin/env python
"""Draw annotation samples from the cleaned master file and write annotator workbooks.

  python scripts/make_annotation_sample.py --stage pilot            # about 40 diverse videos, same set for A and B
  python scripts/make_annotation_sample.py --stage main             # about 400 videos (200 per platform)

Outputs go to data/annotation/<stage>/ :
  <stage>_key.csv               full metadata for every sampled video (KEEP PRIVATE, never give to annotators)
  <stage>_annotator_A.xlsx      what annotator A sees and fills
  <stage>_annotator_B.xlsx      pilot: same videos as A (other order). main: only the double-annotated subset
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.annotation import assign_primary_subtopic, diverse_sample, stratified_sample  # noqa: E402
from vidscreen.annotation.workbook import write_annotator_workbook  # noqa: E402
from vidscreen.config import load_settings  # noqa: E402
from vidscreen.preprocessing.merge import read_csv_str  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "main"], required=True)
    ap.add_argument("--n", type=int, default=40, help="pilot: number of videos (30-50)")
    ap.add_argument("--n-per-platform", type=int, default=200, help="main: videos per platform")
    ap.add_argument("--double-frac", type=float, default=0.20, help="main: share labeled by both annotators")
    ap.add_argument("--min-per-stratum", type=int, default=8, help="main: minimum videos per platform x subtopic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--master", help="Override data/intermediate/master_cleaned.csv")
    ap.add_argument("--force", action="store_true", help="overwrite an existing key (this orphans old labels!)")
    args = ap.parse_args()

    s = load_settings()
    master_path = Path(args.master) if args.master else s.intermediate_dir / "master_cleaned.csv"
    if not master_path.exists():
        print(f"ERROR: {master_path} not found. Run scripts/run_preprocessing.py first.")
        return 1
    out_dir = s.root / "data" / "annotation" / args.stage
    key_path = out_dir / f"{args.stage}_key.csv"
    if key_path.exists() and not args.force:
        print(f"ERROR: {key_path} already exists. Re-drawing would orphan existing labels. Use --force only if you are sure.")
        return 1

    master = read_csv_str(master_path)
    qmap_path = s.intermediate_dir / "query_video_map.csv"
    qmap = read_csv_str(qmap_path) if qmap_path.exists() else None
    master["primary_subtopic"] = assign_primary_subtopic(master, qmap)
    print(f"Master: {len(master)} videos {master['platform'].value_counts().to_dict()}")

    pilot_key = s.root / "data" / "annotation" / "pilot" / "pilot_key.csv"
    pool = master
    if args.stage == "main":
        if pilot_key.exists():
            used = set(read_csv_str(pilot_key)["video_uid"])
            pool = master[~master["video_uid"].isin(used)]
            print(f"Excluded {len(master) - len(pool)} pilot videos from the main pool.")
        else:
            print("WARNING: no pilot key found; pilot videos are not excluded.")
        short = {p: n for p, n in pool["platform"].value_counts().items() if n < args.n_per_platform}
        if short:
            print(f"NOTE: fewer videos than requested on {short}; all of them will be taken.")

    rng = np.random.default_rng(args.seed)
    if args.stage == "pilot":
        sample = diverse_sample(pool, args.n, args.seed)
        prefix = "P"
    else:
        sample = stratified_sample(pool, args.n_per_platform, args.seed, min_each=args.min_per_stratum)
        prefix = "M"

    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)  # ids carry no stratum information
    sample.insert(0, "item_id", [f"{prefix}{i + 1:04d}" for i in range(len(sample))])

    if args.stage == "pilot":
        sample["double"] = 1
    else:
        sample["double"] = 0
        for _, g in sample.groupby("platform"):
            k = int(round(args.double_frac * len(g)))
            sample.loc[rng.choice(g.index.to_numpy(), size=k, replace=False), "double"] = 1

    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(key_path, index=False, encoding="utf-8-sig")
    write_annotator_workbook(sample, out_dir / f"{args.stage}_annotator_A.xlsx", order_seed=args.seed + 1)
    b_rows = sample[sample["double"] == 1]
    write_annotator_workbook(b_rows, out_dir / f"{args.stage}_annotator_B.xlsx", order_seed=args.seed + 2)

    print(f"\n{args.stage}: {len(sample)} videos sampled, {len(b_rows)} for annotator B")
    print(sample.groupby(["platform", "primary_subtopic"]).size().unstack(0).fillna(0).astype(int).to_string())
    print(f"\nKey (private): {key_path}\nWorkbooks:     {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
