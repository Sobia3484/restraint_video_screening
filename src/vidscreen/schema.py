from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class VideoRecord:
    """One row in a per-query CSV. Field order == CSV column order."""

    # --- provenance
    query_id: str = ""
    query_text: str = ""
    platform: str = ""                      # youtube | tiktok
    query_rank: Optional[int] = None        # position within that platform's results
    # --- identity
    video_id: str = ""
    url: str = ""
    title: str = ""                         # TikTok: the caption text is stored here
    description: str = ""                   # TikTok: left empty (single caption field)
    # --- creator
    creator_name: str = ""
    creator_id: str = ""                    # YouTube channel_id / TikTok handle
    creator_url: str = ""
    creator_followers: Optional[int] = None
    # --- video facts
    duration_seconds: Optional[float] = None
    upload_date: str = ""                   # YYYY-MM-DD (UTC)
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None            # TikTok only
    saves: Optional[int] = None             # TikTok only (collect_count)
    hashtags: str = ""                      # pipe-separated, lower-case
    tags: str = ""                          # YouTube tags, pipe-separated
    category: str = ""                      # YouTube category
    language: str = ""                      # platform-reported; often blank
    country_code: str = ""                  # only when the platform reports it
    country_source: str = ""
    # --- transcript
    transcript: str = ""
    transcript_status: str = "not_requested"
    transcript_source: str = ""
    transcript_language: str = ""
    transcript_is_auto: Optional[bool] = None
    transcript_word_count: Optional[int] = None
    # --- housekeeping
    metadata_status: str = ""               # ok | unavailable | failed
    thumbnail_url: str = ""
    retrieved_at_utc: str = ""

    def apply_transcript(self, result) -> None:
        from .utils import word_count

        self.transcript = result.text or ""
        self.transcript_status = result.status
        self.transcript_source = result.source
        self.transcript_language = result.language
        self.transcript_is_auto = result.is_auto
        self.transcript_word_count = word_count(self.transcript) if self.transcript else 0

    def to_row(self) -> dict:
        row = {}
        for name in COLUMNS:
            value = getattr(self, name)
            if value is None:
                row[name] = ""
            elif isinstance(value, bool):
                row[name] = "True" if value else "False"
            else:
                row[name] = value
        return row


COLUMNS = [f.name for f in fields(VideoRecord)]
