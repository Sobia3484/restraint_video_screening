"""Orchestration: for every query -> YouTube + TikTok rows -> one CSV per query."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional, Sequence

from .cache import JsonCache
from .config import Settings
from .csv_io import append_manifest, write_records
from .errors import FatalExtractionError
from .queries import Query
from .schema import VideoRecord
from .transcripts import TikTokCaptionFetcher, TranscriptResult, YouTubeTranscriptFetcher
from .utils import utc_now_iso

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    tqdm = None

log = logging.getLogger("vidscreen")


def _progress(iterable, total: int, desc: str):
    return tqdm(iterable, total=total, desc=desc, leave=False) if tqdm else iterable


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        platforms: Sequence[str] = ("youtube", "tiktok"),
        with_transcripts: bool = True,
        force: bool = False,
        *,
        youtube=None,
        tiktok=None,
        yt_transcripts=None,
        tt_captions=None,
    ):
        self.s = settings
        self.platforms = tuple(platforms)
        self.with_transcripts = with_transcripts
        self.force = force
        cache = JsonCache(settings.cache_dir)
        if "youtube" in self.platforms:
            from .extractors.youtube import YouTubeExtractor

            self.youtube = youtube or YouTubeExtractor(settings, cache)
            self.yt_transcripts = yt_transcripts or YouTubeTranscriptFetcher(
                settings.tr_languages, cache, sleep=settings.tr_yt_sleep,
                block_threshold=settings.tr_block_threshold, workers=settings.tr_yt_workers,
            )
        if "tiktok" in self.platforms:
            from .extractors.tiktok import TikTokExtractor

            self.tiktok = tiktok or TikTokExtractor(settings, cache)
            self.tt_captions = tt_captions or TikTokCaptionFetcher(settings.tr_languages, cache)

    # ------------------------------------------------------------------ public
    def run(self, queries: Iterable[Query]) -> list[dict]:
        queries = list(queries)
        self.s.raw_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for i, query in enumerate(queries, 1):
            final_path = self.s.raw_dir / f"{query.file_stem}.csv"
            if final_path.exists() and not self.force:
                log.info("[%s] (%d/%d) already done -> skipping (%s)", query.id, i, len(queries), final_path.name)
                continue
            log.info("[%s] (%d/%d) '%s'", query.id, i, len(queries), query.text)
            manifest_rows.append(self._run_query(query))
        return manifest_rows

    # ----------------------------------------------------------------- private
    def _run_query(self, query: Query) -> dict:
        started = utc_now_iso()
        notes: list[str] = []
        yt_recs: list[VideoRecord] = []
        tt_recs: list[VideoRecord] = []
        credits_before = getattr(getattr(self, "tiktok", None), "credits_spent", 0)

        if "youtube" in self.platforms:
            try:
                yt_recs = self._youtube(query)
                if self.with_transcripts and self.yt_transcripts.halted:
                    notes.append("youtube_transcripts_paused_ip_block")
            except FatalExtractionError:
                raise
            except Exception as exc:  # keep the run alive; mark the query partial
                log.exception("[%s] YouTube failed", query.id)
                notes.append(f"youtube_failed:{type(exc).__name__}")

        if "tiktok" in self.platforms:
            try:
                tt_recs = self._tiktok(query)
                if self.tiktok.limit_hit:
                    notes.append("tiktok_credit_limit_reached")
            except FatalExtractionError:
                raise
            except Exception as exc:
                log.exception("[%s] TikTok failed", query.id)
                notes.append(f"tiktok_failed:{type(exc).__name__}")

        complete = not notes
        final_path = self.s.raw_dir / f"{query.file_stem}.csv"
        partial_path = self.s.raw_dir / f"{query.file_stem}.partial.csv"
        out_path = final_path if complete else partial_path
        write_records(out_path, yt_recs + tt_recs)
        if complete and partial_path.exists():
            partial_path.unlink()

        row = {
            "run_started_utc": started,
            "query_id": query.id,
            "query_text": query.text,
            "output_file": out_path.name,
            "status": "complete" if complete else "partial",
            "youtube_rows": len(yt_recs),
            "youtube_with_transcript": sum(r.transcript_status == "ok" for r in yt_recs),
            "tiktok_rows": len(tt_recs),
            "tiktok_with_transcript": sum(r.transcript_status == "ok" for r in tt_recs),
            "tiktok_credits_spent": getattr(getattr(self, "tiktok", None), "credits_spent", 0) - credits_before,
            "notes": ";".join(notes),
        }
        append_manifest(self.s.raw_dir / "_manifest.csv", row)
        log.info(
            "[%s] %s -> %s | YouTube %d (%d transcripts) | TikTok %d (%d transcripts)",
            query.id, row["status"], out_path.name, row["youtube_rows"], row["youtube_with_transcript"],
            row["tiktok_rows"], row["tiktok_with_transcript"],
        )
        return row

    def _map(self, fn: Callable, items: list, desc: str) -> list:
        workers = self.s.yt_workers
        if workers <= 1 or len(items) <= 1:
            return [fn(x) for x in _progress(items, len(items), desc)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(_progress(pool.map(fn, items), len(items), desc))

    def _youtube(self, query: Query) -> list[VideoRecord]:
        from .extractors.youtube import youtube_record

        ids = self.youtube.search(query)

        def work(arg) -> VideoRecord:
            rank, vid = arg
            try:
                info = self.youtube.fetch_video(vid)
                rec = youtube_record(query, rank, info)
                if not self.with_transcripts:
                    rec.transcript_status = "not_requested"
                elif info.get("_status") == "ok":
                    rec.apply_transcript(self.yt_transcripts.fetch(vid))
                else:
                    rec.transcript_status = "skipped_video_unavailable"
                return rec
            except Exception as exc:
                log.debug("YouTube video %s failed: %s", vid, exc)
                return youtube_record(query, rank, {"id": vid, "_status": "failed", "_error": type(exc).__name__})

        return self._map(work, list(enumerate(ids, 1)), f"{query.id} youtube")

    def _tiktok(self, query: Query) -> list[VideoRecord]:
        from .extractors.tiktok import parse_aweme

        awemes = self.tiktok.search(query)

        def work(arg) -> VideoRecord:
            rank, aweme = arg
            rec = parse_aweme(query, rank, aweme)
            if self.with_transcripts:
                rec.apply_transcript(self.tt_captions.fetch(rec.video_id, aweme))
            else:
                rec.transcript_status = "not_requested"
            return rec

        return self._map(work, list(enumerate(awemes, 1)), f"{query.id} tiktok")
