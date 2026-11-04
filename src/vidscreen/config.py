from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path
    queries_file: Path
    raw_dir: Path
    cache_dir: Path
    log_dir: Path
    intermediate_dir: Path
    final_dir: Path
    # YouTube (yt-dlp)
    yt_results_per_query: int = 50
    yt_workers: int = 4
    yt_sleep: float = 1.0
    yt_max_retries: int = 3
    yt_cookies_file: str = ""
    # TikTok (ScrapeCreators)
    tt_pages_per_query: int = 2
    tt_date_posted: str = "all-time"
    tt_sort_by: str = "relevance"
    tt_region: str = ""
    tt_max_credits: int = 90
    tt_credit_reserve: int = 5
    tt_sleep: float = 0.5
    tt_timeout: int = 60
    tt_max_retries: int = 3
    tt_api_key: str = field(default="", repr=False)
    # Transcripts
    tr_enabled: bool = True
    tr_languages: tuple = ("en", "ur", "hi")
    tr_block_threshold: int = 5
    tr_yt_sleep: float = 3.0
    tr_yt_workers: int = 1


def load_settings(config_path: Optional[str] = None, root: Optional[Path] = None) -> Settings:
    root = Path(root) if root else PROJECT_ROOT
    if load_dotenv is not None:
        load_dotenv(root / ".env")
    cfg_file = Path(config_path) if config_path else root / "config" / "settings.yaml"
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    paths, yt, tt, tr = (cfg.get(k) or {} for k in ("paths", "youtube", "tiktok", "transcripts"))

    def p(rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else root / path

    return Settings(
        root=root,
        queries_file=p(paths.get("queries_file", "config/queries.csv")),
        raw_dir=p(paths.get("raw_dir", "data/raw")),
        cache_dir=p(paths.get("cache_dir", "data/cache")),
        log_dir=p(paths.get("log_dir", "logs")),
        intermediate_dir=p(paths.get("intermediate_dir", "data/intermediate")),
        final_dir=p(paths.get("final_dir", "data/final")),
        yt_results_per_query=int(yt.get("results_per_query", 50)),
        yt_workers=max(1, int(yt.get("workers", 4))),
        yt_sleep=float(yt.get("request_sleep_seconds", 1.0)),
        yt_max_retries=int(yt.get("max_retries", 3)),
        yt_cookies_file=str(yt.get("cookies_file", "") or ""),
        tt_pages_per_query=int(tt.get("pages_per_query", 2)),
        tt_date_posted=str(tt.get("date_posted", "all-time")),
        tt_sort_by=str(tt.get("sort_by", "relevance")),
        tt_region=str(tt.get("region", "") or ""),
        tt_max_credits=int(tt.get("max_credits_per_run", 90)),
        tt_credit_reserve=int(tt.get("credit_reserve", 5)),
        tt_sleep=float(tt.get("request_sleep_seconds", 0.5)),
        tt_timeout=int(tt.get("timeout_seconds", 60)),
        tt_max_retries=int(tt.get("max_retries", 3)),
        tt_api_key=os.getenv("SCRAPECREATORS_API_KEY", "").strip(),
        tr_enabled=bool(tr.get("enabled", True)),
        tr_languages=tuple(tr.get("preferred_languages", ["en", "ur", "hi"])),
        tr_block_threshold=int(tr.get("block_threshold", 5)),
        tr_yt_sleep=float(tr.get("youtube_sleep_seconds", 3.0)),
        tr_yt_workers=max(1, int(tr.get("youtube_workers", 1))),
    )
