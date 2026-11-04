"""TikTok extraction through the ScrapeCreators keyword-search endpoint.

GET https://api.scrapecreators.com/v1/tiktok/search/keyword
    params: query, date_posted, sort_by, region, cursor      header: x-api-key
Each page costs 1 credit. Every raw page is cached on disk, so re-runs never re-spend credits.
"""
from __future__ import annotations

import logging

import requests

from ..cache import JsonCache
from ..config import Settings
from ..errors import CreditLimitReached, ExtractionError, FatalExtractionError
from ..queries import Query
from ..schema import VideoRecord
from ..utils import extract_hashtags, join_pipe, polite_sleep, retry, to_int, unix_to_date, utc_now_iso

log = logging.getLogger("vidscreen")
API_BASE = "https://api.scrapecreators.com"


class _Retryable(Exception):
    pass


class TikTokExtractor:
    def __init__(self, settings: Settings, cache: JsonCache, session: requests.Session | None = None):
        self.s = settings
        self.cache = cache
        self.session = session or requests.Session()
        self.session.headers.update({"x-api-key": settings.tt_api_key, "Accept": "application/json"})
        self.credits_spent = 0
        self.credits_remaining: int | None = None
        self.limit_hit = False

    # ------------------------------------------------------------------ public
    def search(self, query: Query) -> list[dict]:
        """Return de-duplicated `aweme_info` dicts for a query (up to pages_per_query pages)."""
        self.limit_hit = False
        items, seen, cursor = [], set(), None
        for _ in range(self.s.tt_pages_per_query):
            try:
                data, from_cache = self._get_page(query, cursor)
            except CreditLimitReached as exc:
                self.limit_hit = True
                log.warning("[%s] TikTok stopped: %s", query.id, exc)
                break
            page = data.get("search_item_list") or []
            for entry in page:
                aweme = (entry or {}).get("aweme_info")
                vid = str((aweme or {}).get("aweme_id") or "")
                if vid and vid not in seen:
                    seen.add(vid)
                    items.append(aweme)
            nxt = data.get("cursor")
            log.debug("[%s] TikTok page cursor=%s -> %d items (cache=%s)", query.id, cursor, len(page), from_cache)
            if not page or nxt is None or nxt == cursor:
                break
            cursor = nxt
        log.info("[%s] TikTok search: %d videos (credits spent this run: %d)", query.id, len(items), self.credits_spent)
        return items

    # ----------------------------------------------------------------- private
    def _get_page(self, query: Query, cursor):
        key = f"{query.id}_{self.s.tt_date_posted}_{self.s.tt_sort_by}_{self.s.tt_region or 'any'}_c{cursor or 0}"
        cached = self.cache.get("tt_search", key)
        if cached is not None:
            return cached, True
        if self.s.tt_max_credits and self.credits_spent >= self.s.tt_max_credits:
            raise CreditLimitReached(f"per-run credit budget reached ({self.s.tt_max_credits})")
        if self.credits_remaining is not None and self.credits_remaining <= self.s.tt_credit_reserve:
            raise CreditLimitReached(f"account credits down to reserve ({self.credits_remaining} left)")
        params = {"query": query.text, "date_posted": self.s.tt_date_posted, "sort_by": self.s.tt_sort_by}
        if self.s.tt_region:
            params["region"] = self.s.tt_region
        if cursor is not None:
            params["cursor"] = cursor
        data = self._request(params)
        self.credits_spent += to_int(data.get("credits_charged")) or 1
        remaining = to_int(data.get("credits_remaining"))
        if remaining is not None:
            self.credits_remaining = remaining
        self.cache.set("tt_search", key, data)
        polite_sleep(self.s.tt_sleep)
        return data, False

    def _request(self, params: dict) -> dict:
        url = f"{API_BASE}/v1/tiktok/search/keyword"

        def _do():
            resp = self.session.get(url, params=params, timeout=self.s.tt_timeout)
            if resp.status_code == 401:
                raise FatalExtractionError("ScrapeCreators rejected the API key (HTTP 401). Check .env.")
            if resp.status_code == 402:
                raise CreditLimitReached("ScrapeCreators reports no credits left (HTTP 402)")
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _Retryable(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()

        data = retry(
            _do,
            attempts=self.s.tt_max_retries,
            base_delay=3.0,
            retry_on=(_Retryable, requests.ConnectionError, requests.Timeout),
        )
        if isinstance(data, dict) and data.get("success") is False:
            raise ExtractionError(f"ScrapeCreators returned success=false: {str(data)[:200]}")
        return data


# --------------------------------------------------------------------- parsing
def _video_url(aweme: dict, handle: str, vid: str) -> str:
    if handle and vid:
        return f"https://www.tiktok.com/@{handle}/video/{vid}"
    share = (aweme.get("share_url") or "").split("?")[0]
    return share or aweme.get("url") or ""


def _duration_seconds(video: dict):
    ms = to_int(video.get("duration"))  # TikTok reports video duration in milliseconds
    return None if ms is None else round(ms / 1000.0, 2)


def parse_aweme(query: Query, rank: int, aweme: dict) -> VideoRecord:
    """Map one ScrapeCreators `aweme_info` object to a VideoRecord."""
    author = aweme.get("author") or {}
    stats = aweme.get("statistics") or {}
    video = aweme.get("video") or {}
    vid = str(aweme.get("aweme_id") or "")
    handle = author.get("unique_id") or ""
    desc = (aweme.get("desc") or "").strip()

    tags = [t.get("hashtag_name") for t in aweme.get("text_extra") or [] if t.get("hashtag_name")]
    tags += [c.get("cha_name") for c in aweme.get("cha_list") or [] if c.get("cha_name")]
    tags += extract_hashtags(desc)
    seen, hashtags = set(), []
    for t in tags:
        t = t.lower()
        if t not in seen:
            seen.add(t)
            hashtags.append(t)

    region = aweme.get("region") or ""
    covers = (video.get("cover") or {}).get("url_list") or []
    return VideoRecord(
        query_id=query.id,
        query_text=query.text,
        platform="tiktok",
        query_rank=rank,
        video_id=vid,
        url=_video_url(aweme, handle, vid),
        title=desc,
        creator_name=author.get("nickname") or "",
        creator_id=handle,
        creator_url=f"https://www.tiktok.com/@{handle}" if handle else "",
        creator_followers=to_int(author.get("follower_count")),
        duration_seconds=_duration_seconds(video),
        upload_date=unix_to_date(aweme.get("create_time")) or (aweme.get("create_time_utc") or "")[:10],
        views=to_int(stats.get("play_count")),
        likes=to_int(stats.get("digg_count")),
        comments=to_int(stats.get("comment_count")),
        shares=to_int(stats.get("share_count")),
        saves=to_int(stats.get("collect_count")),
        hashtags=join_pipe(hashtags),
        language=aweme.get("desc_language") or "",
        country_code=region,
        country_source="tiktok_video_region" if region else "",
        metadata_status="ok",
        thumbnail_url=covers[0] if covers else "",
        retrieved_at_utc=utc_now_iso(),
    )
