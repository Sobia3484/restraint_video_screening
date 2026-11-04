"""YouTube extraction with yt-dlp: search -> video ids -> full metadata (all cached)."""
from __future__ import annotations

import logging
import re

from ..cache import JsonCache
from ..config import Settings
from ..queries import Query
from ..schema import VideoRecord
from ..utils import extract_hashtags, join_pipe, polite_sleep, retry, to_int, utc_now_iso, yyyymmdd_to_iso

log = logging.getLogger("vidscreen")

KEEP_KEYS = (
    "id", "title", "description", "uploader", "channel", "channel_id", "channel_url",
    "channel_follower_count", "duration", "view_count", "like_count", "comment_count",
    "upload_date", "timestamp", "tags", "categories", "language", "thumbnail",
    "webpage_url", "live_status", "availability", "age_limit",
)
_PERMANENT_MARKERS = (
    "video unavailable", "private video", "has been removed", "this video is not available",
    "account associated with this video has been terminated", "members-only", "confirm your age",
    "copyright", "no longer available", "please sign in",
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _QuietLogger:
    """Route yt-dlp messages to our debug log instead of spamming the console."""

    def debug(self, msg): log.debug("yt-dlp: %s", msg)
    info = debug
    warning = debug
    def error(self, msg): log.debug("yt-dlp error: %s", _ANSI_RE.sub("", str(msg))[:200])


def _clean_err(exc: Exception) -> str:
    return _ANSI_RE.sub("", str(exc)).strip()[:200]


def _is_permanent(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _PERMANENT_MARKERS)


class YouTubeExtractor:
    def __init__(self, settings: Settings, cache: JsonCache):
        self.s = settings
        self.cache = cache

    def _base_opts(self) -> dict:
        opts = {"quiet": True, "no_warnings": True, "noprogress": True, "skip_download": True, "socket_timeout": 30, "logger": _QuietLogger()}
        if self.s.yt_cookies_file:
            opts["cookiefile"] = self.s.yt_cookies_file
        return opts

    def search(self, query: Query) -> list[str]:
        """Return an ordered, de-duplicated list of video ids for a query."""
        n = self.s.yt_results_per_query
        key = f"{query.id}_n{n}"
        cached = self.cache.get("yt_search", key)
        if cached is not None:
            return cached["ids"]
        import yt_dlp

        opts = {**self._base_opts(), "extract_flat": "in_playlist"}

        def _do():
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(f"ytsearch{n}:{query.text}", download=False)

        result = retry(_do, attempts=self.s.yt_max_retries, base_delay=3.0)
        ids, seen = [], set()
        for entry in (result or {}).get("entries") or []:
            vid = (entry or {}).get("id")
            if vid and vid not in seen:
                seen.add(vid)
                ids.append(vid)
        self.cache.set("yt_search", key, {"query": query.text, "ids": ids, "fetched_at": utc_now_iso()})
        log.info("[%s] YouTube search: %d videos", query.id, len(ids))
        return ids

    def fetch_video(self, video_id: str) -> dict:
        """Return a metadata dict with `_status` in {ok, unavailable, failed}."""
        cached = self.cache.get("yt_video", video_id)
        if cached is not None:
            return cached
        polite_sleep(self.s.yt_sleep)
        import yt_dlp
        from yt_dlp.utils import DownloadError

        opts = {**self._base_opts(), "noplaylist": True}
        url = f"https://www.youtube.com/watch?v={video_id}"

        def _do():
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)

        try:
            info = retry(
                _do,
                attempts=self.s.yt_max_retries,
                base_delay=3.0,
                retry_on=(DownloadError,),
                should_retry=lambda e: not _is_permanent(e),
            )
        except DownloadError as exc:
            status = "unavailable" if _is_permanent(exc) else "failed"
            data = {"id": video_id, "_status": status, "_error": _clean_err(exc)}
            if status == "unavailable":
                self.cache.set("yt_video", video_id, data)
            return data
        except Exception as exc:
            return {"id": video_id, "_status": "failed", "_error": f"{type(exc).__name__}: {_clean_err(exc)}"[:200]}
        data = {k: info.get(k) for k in KEEP_KEYS}
        data["id"] = data.get("id") or video_id
        data["_status"] = "ok"
        self.cache.set("yt_video", video_id, data)
        return data


def youtube_record(query: Query, rank: int, info: dict) -> VideoRecord:
    """Map a cached yt-dlp metadata dict to a VideoRecord."""
    vid = info.get("id") or ""
    rec = VideoRecord(
        query_id=query.id,
        query_text=query.text,
        platform="youtube",
        query_rank=rank,
        video_id=vid,
        url=f"https://www.youtube.com/watch?v={vid}",
        retrieved_at_utc=utc_now_iso(),
    )
    status = info.get("_status", "failed")
    if status != "ok":
        rec.metadata_status = status if not info.get("_error") else f"{status}: {info['_error']}"
        return rec
    title, description = info.get("title") or "", info.get("description") or ""
    rec.metadata_status = "ok"
    rec.title = title
    rec.description = description
    rec.creator_name = info.get("channel") or info.get("uploader") or ""
    rec.creator_id = info.get("channel_id") or ""
    rec.creator_url = info.get("channel_url") or ""
    rec.creator_followers = to_int(info.get("channel_follower_count"))
    rec.duration_seconds = to_int(info.get("duration"))
    rec.upload_date = yyyymmdd_to_iso(info.get("upload_date"))
    rec.views = to_int(info.get("view_count"))
    rec.likes = to_int(info.get("like_count"))
    rec.comments = to_int(info.get("comment_count"))
    rec.hashtags = join_pipe(extract_hashtags(title, description))
    rec.tags = join_pipe(info.get("tags") or [])
    rec.category = join_pipe(info.get("categories") or [])
    rec.language = info.get("language") or ""
    rec.thumbnail_url = info.get("thumbnail") or ""
    return rec
