from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_settings
from .errors import FatalExtractionError
from .logging_setup import setup_logging
from .pipeline import Pipeline
from .queries import load_queries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract YouTube + TikTok videos for each search query (one CSV per query).")
    p.add_argument("--platform", choices=["both", "youtube", "tiktok"], default="both")
    p.add_argument("--queries", nargs="*", metavar="QID", help="Only these query ids, e.g. Q001 Q014")
    p.add_argument("--limit", type=int, help="Only the first N queries")
    p.add_argument("--force", action="store_true", help="Rebuild CSVs that already exist (cached data is reused)")
    p.add_argument("--no-transcripts", action="store_true", help="Skip transcript extraction")
    p.add_argument("--yt-results", type=int, help="Override YouTube results per query")
    p.add_argument("--tt-pages", type=int, help="Override TikTok pages per query (1 page = 1 credit)")
    p.add_argument("--output-dir", help="Write CSVs here instead of data/raw")
    p.add_argument("--config", help="Path to a settings.yaml")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    overrides = {}
    if args.yt_results:
        overrides["yt_results_per_query"] = args.yt_results
    if args.tt_pages:
        overrides["tt_pages_per_query"] = args.tt_pages
    if args.output_dir:
        out = Path(args.output_dir)
        overrides["raw_dir"] = out if out.is_absolute() else settings.root / out
    settings = replace(settings, **overrides)

    log_path = setup_logging(settings.log_dir, args.log_level)
    log = logging.getLogger("vidscreen")

    queries = load_queries(settings.queries_file)
    if args.queries:
        wanted = {q.upper() for q in args.queries}
        queries = [q for q in queries if q.id.upper() in wanted]
        missing = wanted - {q.id.upper() for q in queries}
        if missing:
            log.error("Unknown query ids: %s", ", ".join(sorted(missing)))
            return 2
    if args.limit:
        queries = queries[: args.limit]

    platforms = ("youtube", "tiktok") if args.platform == "both" else (args.platform,)
    if "tiktok" in platforms and not settings.tt_api_key:
        log.error("SCRAPECREATORS_API_KEY is missing. Copy .env.example to .env and add your key "
                  "(or run with --platform youtube).")
        return 2

    pipeline = Pipeline(settings, platforms, with_transcripts=settings.tr_enabled and not args.no_transcripts, force=args.force)
    log.info("Log file: %s", log_path)
    try:
        rows = pipeline.run(queries)
    except FatalExtractionError as exc:
        log.error("Fatal: %s", exc)
        return 1

    done = sum(r["status"] == "complete" for r in rows)
    partial = [r["query_id"] for r in rows if r["status"] != "complete"]
    log.info("Finished: %d complete, %d partial%s", done, len(partial), f" ({', '.join(partial)})" if partial else "")
    if getattr(pipeline, "tiktok", None) is not None:
        log.info("TikTok credits spent this run: %d (remaining reported by API: %s)",
                 pipeline.tiktok.credits_spent, pipeline.tiktok.credits_remaining)
    return 0 if not partial else 1


if __name__ == "__main__":
    sys.exit(main())
