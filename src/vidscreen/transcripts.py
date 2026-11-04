"""Transcript acquisition.

* YouTube : youtube-transcript-api (existing captions, manual preferred over auto).
* TikTok  : auto-caption WebVTT URL that ScrapeCreators already returns inside the
            search response (video.cla_info.caption_infos). Downloading it costs 0 credits.
Speech-to-text (faster-whisper) is intentionally NOT run here; it is a later fallback
for videos whose status is no_transcript / no_caption_available.
"""
from __future__ import annotations

import html
import logging
import re
import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

from .cache import JsonCache

log = logging.getLogger("vidscreen")


@dataclass
class TranscriptResult:
    text: str = ""
    status: str = "not_requested"
    source: str = ""
    language: str = ""
    is_auto: Optional[bool] = None


# ----------------------------------------------------------------------------- WebVTT
def parse_webvtt(raw: str) -> str:
    """Convert WebVTT text to a single plain-text string."""
    lines = [ln.strip() for ln in (raw or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out, prev = [], None
    for i, line in enumerate(lines):
        if not line or "-->" in line:
            continue
        if line.startswith(("WEBVTT", "NOTE", "STYLE", "Kind:", "Language:")):
            continue
        if i + 1 < len(lines) and "-->" in lines[i + 1]:  # cue identifier line
            continue
        line = html.unescape(re.sub(r"<[^>]+>", "", line)).strip()
        if line and line != prev:
            out.append(line)
            prev = line
    return " ".join(out)


# ----------------------------------------------------------------------------- YouTube
_PERMANENT = {"ok", "no_transcript", "disabled", "video_unavailable", "age_restricted"}


def _map_error(exc: Exception) -> str:
    name = type(exc).__name__
    table = {
        "TranscriptsDisabled": "disabled",
        "NoTranscriptFound": "no_transcript",
        "NoTranscriptAvailable": "no_transcript",
        "VideoUnavailable": "video_unavailable",
        "VideoUnplayable": "video_unavailable",
        "AgeRestricted": "age_restricted",
        "RequestBlocked": "blocked",
        "IpBlocked": "blocked",
    }
    return table.get(name, f"error:{name}")


class YouTubeTranscriptFetcher:
    def __init__(self, languages, cache: JsonCache, sleep: float = 3.0, block_threshold: int = 5, workers: int = 1):
        self.languages = list(languages)
        self.cache = cache
        self.sleep = sleep
        self.block_threshold = block_threshold
        self.halted = False
        self._blocks = 0
        self._lock = threading.Lock()
        # YouTube blocks IPs that request transcripts too fast, so cap concurrency independently
        # of the metadata workers.
        self._slots = threading.BoundedSemaphore(max(1, workers))
        # Minimum gap between requests, shared by all threads (see _throttle).
        self._next_ok = 0.0
        self._rate_lock = threading.Lock()

    def fetch(self, video_id: str) -> TranscriptResult:
        cached = self.cache.get("yt_transcript", video_id)
        if cached is not None:
            return TranscriptResult(**cached)
        if self.halted:
            return TranscriptResult(status="skipped_blocked")
        with self._slots:
            if self.halted:
                return TranscriptResult(status="skipped_blocked")
            self._throttle()
            result = self._fetch_uncached(video_id)
        with self._lock:
            if result.status == "blocked":
                self._blocks += 1
                if self._blocks >= self.block_threshold and not self.halted:
                    self.halted = True
                    log.warning(
                        "YouTube is blocking transcript requests from this IP (%d in a row). "
                        "Transcripts are paused for this run; re-run later or from another network.",
                        self._blocks,
                    )
            else:
                self._blocks = 0
        if result.status in _PERMANENT:
            self.cache.set("yt_transcript", video_id, asdict(result))
        return result

    def reset_halt(self) -> None:
        """Resume after a cooldown (clears the 'blocked' pause)."""
        with self._lock:
            self.halted = False
            self._blocks = 0

    def _throttle(self) -> None:
        with self._rate_lock:
            wait = self._next_ok - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next_ok = time.monotonic() + self.sleep * random.uniform(0.8, 1.4)

    def _choose(self, transcript_list):
        for finder in (transcript_list.find_manually_created_transcript, transcript_list.find_generated_transcript):
            try:
                return finder(self.languages)
            except Exception:
                continue
        for transcript in transcript_list:  # any other language
            return transcript
        return None

    def _fetch_uncached(self, video_id: str) -> TranscriptResult:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return TranscriptResult(status="error:youtube_transcript_api_not_installed")
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            chosen = self._choose(transcript_list)
            if chosen is None:
                return TranscriptResult(status="no_transcript")
            segments = chosen.fetch().to_raw_data()
            text = " ".join(str(s.get("text", "")).replace("\n", " ").strip() for s in segments).strip()
            return TranscriptResult(
                text=text,
                status="ok" if text else "no_transcript",
                source="youtube_captions",
                language=getattr(chosen, "language_code", "") or "",
                is_auto=bool(getattr(chosen, "is_generated", False)),
            )
        except Exception as exc:  # library raises many specific subclasses
            return TranscriptResult(status=_map_error(exc))


# ----------------------------------------------------------------------------- TikTok
def caption_infos(aweme: dict) -> list:
    return (((aweme.get("video") or {}).get("cla_info") or {}).get("caption_infos")) or []


def pick_caption(infos: list, languages) -> dict:
    """Prefer original-language captions, then the configured language order."""
    langs = [l.lower() for l in languages]

    def key(c):
        code = (c.get("language_code") or "").lower()
        rank = langs.index(code) if code in langs else len(langs)
        return (0 if c.get("is_original_caption") else 1, rank)

    return sorted(infos, key=key)[0]


class TikTokCaptionFetcher:
    def __init__(self, languages, cache: JsonCache, timeout: int = 30):
        self.languages = list(languages)
        self.cache = cache
        self.timeout = timeout

    def fetch(self, video_id: str, aweme: dict) -> TranscriptResult:
        cached = self.cache.get("tt_caption", video_id)
        if cached is not None:
            return TranscriptResult(**cached)
        infos = caption_infos(aweme)
        if not infos:
            result = TranscriptResult(status="no_caption_available")
            self.cache.set("tt_caption", video_id, asdict(result))
            return result
        chosen = pick_caption(infos, self.languages)
        url = chosen.get("url") or next(iter(chosen.get("url_list") or []), "")
        if not url:
            return TranscriptResult(status="no_caption_available")
        try:
            import requests

            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            # resp.text would guess ISO-8859-1 for text/vtt and garble non-English captions
            text = parse_webvtt(resp.content.decode("utf-8-sig", errors="replace"))
        except Exception as exc:
            log.debug("TikTok caption download failed for %s: %s", video_id, exc)
            return TranscriptResult(status="caption_download_failed")
        result = TranscriptResult(
            text=text,
            status="ok" if text else "empty_caption",
            source="tiktok_auto_caption" if chosen.get("is_auto_generated") else "tiktok_caption",
            language=chosen.get("language_code") or "",
            is_auto=chosen.get("is_auto_generated"),
        )
        self.cache.set("tt_caption", video_id, asdict(result))
        return result
