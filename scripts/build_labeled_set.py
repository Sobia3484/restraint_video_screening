#!/usr/bin/env python
"""Merge annotator labels and adjudication into the final labeled set.

  python scripts/build_labeled_set.py

Reads  data/annotation/main/main_key.csv, main_annotator_A.xlsx, main_annotator_B.xlsx,
       main_disagreements.csv (final_label filled by the adjudicator)
Writes data/final/labeled_set.csv
Rules: double items that agree keep that label; disagreements and any 'Unsure' need the
adjudicator's final_label; single items keep annotator A's label.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.annotation.agreement import LABELS  # noqa: E402
from vidscreen.annotation.workbook import read_annotations  # noqa: E402
from vidscreen.config import load_settings  # noqa: E402
from vidscreen.preprocessing.merge import read_csv_str  # noqa: E402


def read_adjudication(path: Path) -> pd.DataFrame:
    """Read the adjudication CSV. Excel's plain 'CSV' is cp1252, 'CSV UTF-8' is utf-8: accept both.
    Label spelling is normalised (relevant -> Relevant)."""
    df = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        for sep in (",", ";", "\t"):  # Excel uses ; or tab in some regional settings
            try:
                cand = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc, sep=sep)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            if {"item_id", "final_label"}.issubset(cand.columns):
                df = cand
                break
        if df is not None:
            break
    if df is None:
        raise ValueError(f"Cannot read {path.name} (encoding or separator). Re-save it as CSV UTF-8 from Excel.")
    missing = [c for c in ("item_id", "final_label", "adjudicator") if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    canon = {l.lower(): l for l in LABELS}
    df["item_id"] = df["item_id"].str.strip()
    df["final_label"] = df["final_label"].str.strip().str.lower().map(canon).fillna("")
    return df[["item_id", "final_label", "adjudicator"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true", help="keep items without an adjudicated label (label_final blank)")
    args = ap.parse_args()

    s = load_settings()
    d = s.root / "data" / "annotation" / "main"
    key = read_csv_str(d / "main_key.csv")
    a = read_annotations(d / "main_annotator_A.xlsx")
    b = read_annotations(d / "main_annotator_B.xlsx")
    adj_path = d / "main_disagreements.csv"
    adj = read_adjudication(adj_path) if adj_path.exists() else pd.DataFrame(columns=["item_id", "final_label", "adjudicator"])

    miss_a = sorted(set(key["item_id"]) - set(a.loc[a["label"].isin(LABELS), "item_id"]))
    if miss_a:
        print(f"ERROR: annotator A has no valid label for {len(miss_a)} items (e.g. {miss_a[:5]}).")
        return 1
    dbl = set(key.loc[key["double"] == "1", "item_id"])
    miss_b = sorted(dbl - set(b.loc[b["label"].isin(LABELS), "item_id"]))
    if miss_b:
        print(f"ERROR: annotator B has no valid label for {len(miss_b)} double items (e.g. {miss_b[:5]}).")
        return 1

    ren = lambda df, x: df.rename(columns={c: f"{c}_{x}" for c in df.columns if c != "item_id"})  # noqa: E731
    m = key.merge(ren(a, "a"), on="item_id", how="left").merge(ren(b, "b"), on="item_id", how="left").fillna({"label_b": ""})
    m = m.merge(adj[["item_id", "final_label", "adjudicator"]], on="item_id", how="left").fillna({"final_label": "", "adjudicator": ""})

    both = m["label_b"] != ""
    needs = (both & (m["label_a"] != m["label_b"])) | (m["label_a"] == "Unsure") | (m["label_b"] == "Unsure")
    m["label_source"] = "single"
    m.loc[both & ~needs, "label_source"] = "agreed"
    m.loc[needs, "label_source"] = "adjudicated"
    m["label_final"] = m["label_a"]
    m.loc[needs, "label_final"] = m.loc[needs, "final_label"]
    bad = needs & ~m["label_final"].isin(LABELS)
    if bad.any():
        print(f"{int(bad.sum())} items still need a valid final_label in {adj_path.name} (e.g. {m.loc[bad, 'item_id'].head(5).tolist()}).")
        if not args.allow_missing:
            return 1
        m.loc[bad, "label_final"] = ""
    m["unsafe_demo"] = (m["unsafe_demo_a"].eq("Yes") | m["unsafe_demo_b"].eq("Yes")).map({True: "Yes", False: "No"})
    m["use_for_model"] = m["label_final"].isin(["Relevant", "Irrelevant"])

    s.final_dir.mkdir(parents=True, exist_ok=True)
    out = s.final_dir / "labeled_set.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {out} ({len(m)} videos)")
    print(m.groupby(["platform", "label_final"]).size().unstack(1).fillna(0).astype(int).to_string())
    print("\nlabel_source:", m["label_source"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())