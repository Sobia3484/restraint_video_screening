#!/usr/bin/env python
"""Exploratory data analysis of the screened videos.

    python scripts/run_eda.py                                   # strict and all datasets, every section
    python scripts/run_eda.py --dataset strict --sections duration views

Reads  : data/final/screening/relevant_strict_videos.csv   (dataset "strict")
         + audit_videos.csv and irrelevant_videos.csv       (dataset "all" = all three groups)
Writes : data/final/eda/features_<dataset>.csv, reports/eda/<dataset>/ (EDA_report.md, tables/, figures/)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.config import load_settings  # noqa: E402
from vidscreen.eda.runner import DEFAULT_SECTIONS, run_eda  # noqa: E402
from vidscreen.eda.sections import SECTIONS  # noqa: E402
from vidscreen.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["strict", "all", "both"], default="both")
    ap.add_argument("--sections", nargs="*", default=DEFAULT_SECTIONS, help="available: " + ", ".join(SECTIONS))
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    s = load_settings()
    setup_logging(s.log_dir)
    ds = ("strict", "all") if a.dataset == "both" else (a.dataset,)
    done = run_eda(s, ds, a.sections, a.out_dir)
    if not done:
        print("Nothing was run. Check that the screening files exist in data/final/screening/.")
        return 1
    for k, v in done.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
