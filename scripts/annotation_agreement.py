#!/usr/bin/env python
"""Agreement between annotators A and B, and the list for the third adjudicator.

  python scripts/annotation_agreement.py --stage pilot
  python scripts/annotation_agreement.py --stage main

Reads data/annotation/<stage>/<stage>_annotator_A.xlsx and _B.xlsx (filled in), writes
  <stage>_agreement_report.json
  <stage>_disagreements.csv      disagreements + every 'Unsure'; the adjudicator fills final_label
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.annotation import build_adjudication_list, compare_annotators  # noqa: E402
from vidscreen.annotation.workbook import read_annotations  # noqa: E402
from vidscreen.config import load_settings  # noqa: E402
from vidscreen.preprocessing.merge import read_csv_str  # noqa: E402

KAPPA_TARGET = 0.70


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "main"], required=True)
    ap.add_argument("--a", help="Override path of annotator A workbook")
    ap.add_argument("--b", help="Override path of annotator B workbook")
    args = ap.parse_args()

    s = load_settings()
    d = s.root / "data" / "annotation" / args.stage
    key = read_csv_str(d / f"{args.stage}_key.csv")
    a = read_annotations(Path(args.a) if args.a else d / f"{args.stage}_annotator_A.xlsx")
    b = read_annotations(Path(args.b) if args.b else d / f"{args.stage}_annotator_B.xlsx")

    # B only labels the double subset; compare on those items
    a_common = a[a["item_id"].isin(b["item_id"])]
    rep = compare_annotators(a_common, b, key.set_index("item_id")["platform"])
    adj = build_adjudication_list(key[key["double"] == "1"] if args.stage == "main" else key, a, b)
    if args.stage == "main":  # items only A labeled: only the 'Unsure' ones need a third look
        single = build_adjudication_list(key[key["double"] != "1"], a, None)
        adj = pd.concat([adj, single], ignore_index=True)

    (d / f"{args.stage}_agreement_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    adj.to_csv(d / f"{args.stage}_disagreements.csv", index=False, encoding="utf-8-sig")

    print(f"Items compared:        {rep['n_items']}")
    print(f"Percent agreement:     {rep['percent_agreement']}%")
    print(f"Cohen's kappa (3 lbl): {rep['kappa_3class']}   (target >= {KAPPA_TARGET})")
    print(f"Kappa Relevant vs rest:{rep['kappa_relevant_vs_rest']}")
    print(f"Confusion (rows=A, cols=B, order {rep['label_order']}): {rep['confusion_A_rows_B_cols']}")
    for p, v in rep.get("by_platform", {}).items():
        print(f"  {p:8s} n={v['n']} kappa={v['kappa_3class']}")
    print(f"\nFor adjudication: {len(adj)} items -> {d / (args.stage + '_disagreements.csv')}")
    if args.stage == "pilot":
        ok = rep["kappa_3class"] >= KAPPA_TARGET
        print("PILOT " + ("PASSED: you can start the main annotation." if ok else
              "BELOW TARGET: discuss the disagreements, refine docs/screening_criteria.md, repeat the pilot with a new set (--force, new --seed)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
