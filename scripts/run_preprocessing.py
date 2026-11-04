#!/usr/bin/env python
"""Stage 2:  python scripts/run_preprocessing.py [--step merge|dedupe|clean|all]

raw/Q*.csv -> intermediate/master_raw.csv -> master_deduplicated.csv (+ query_video_map.csv)
           -> master_cleaned.csv (+ dropped_records.csv, preprocessing_report.json)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.config import load_settings  # noqa: E402
from vidscreen.preprocessing import build_query_map, clean_master, deduplicate, merge_raw  # noqa: E402
from vidscreen.preprocessing.merge import read_csv_str  # noqa: E402
from vidscreen.queries import load_queries  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["merge", "dedupe", "clean", "all"], default="all")
    ap.add_argument("--raw-dir", help="Override data/raw")
    ap.add_argument("--out-dir", help="Override data/intermediate")
    args = ap.parse_args()

    s = load_settings()
    raw_dir = Path(args.raw_dir) if args.raw_dir else s.raw_dir
    out = Path(args.out_dir) if args.out_dir else s.intermediate_dir
    out.mkdir(parents=True, exist_ok=True)
    queries = load_queries(s.queries_file)
    meta = {q.id: q.subtopic for q in queries}
    report_path = out / "preprocessing_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    raw_path, dedup_path, clean_path = out / "master_raw.csv", out / "master_deduplicated.csv", out / "master_cleaned.csv"
    todo = {"merge": ["merge"], "dedupe": ["dedupe"], "clean": ["clean"], "all": ["merge", "dedupe", "clean"]}[args.step]

    if "merge" in todo:
        master, report["merge"] = merge_raw(raw_dir, raw_path, [q.id for q in queries])
        m = report["merge"]
        print(f"[merge ] {m['files_merged']} files -> {m['rows']} rows {m['rows_by_platform']}")
        if m["missing_queries"]:
            print(f"         WARNING missing queries: {m['missing_queries']}")
    if "dedupe" in todo:
        master = read_csv_str(raw_path)
        dedup, report["dedupe"] = deduplicate(master, meta)
        dedup.to_csv(dedup_path, index=False, encoding="utf-8-sig")
        build_query_map(master, meta).to_csv(out / "query_video_map.csv", index=False, encoding="utf-8-sig")
        d = report["dedupe"]
        print(f"[dedupe] {d['rows_in']} rows -> {d['unique_videos']} unique videos ({d['duplicates_removed']} duplicates removed)")
        for plat, v in d["by_platform"].items():
            print(f"         {plat:8s} {v['rows_in']:5d} -> {v['unique_videos']:5d}  (in 2+ queries: {v['videos_found_by_2plus_queries']})")
    if "clean" in todo:
        dedup = read_csv_str(dedup_path)
        cleaned, dropped, report["clean"] = clean_master(dedup)
        cleaned.to_csv(clean_path, index=False, encoding="utf-8-sig")
        dropped.to_csv(out / "dropped_records.csv", index=False, encoding="utf-8-sig")
        c = report["clean"]
        print(f"[clean ] kept {c['rows_kept']} / {c['rows_in']}; dropped {c['rows_dropped']} {c['dropped_by_reason']}")
        for plat, v in c["by_platform"].items():
            print(f"         {plat:8s} videos={v['videos']}  transcript={v['has_transcript_pct']}%  country_unknown={v['country_unknown_pct']}%")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
