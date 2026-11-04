"""Cleaning rules. Rows are only dropped for the documented reasons; every drop is logged."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

TEXT_COLS = ["title", "description", "creator_name", "creator_id", "hashtags", "tags", "category", "transcript"]
INT_COLS = ["creator_followers", "views", "likes", "comments", "shares", "saves"]
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_YT_URL = re.compile(r"^https://www\.youtube\.com/watch\?v=[\w-]{11}$")
_TT_URL = re.compile(r"^https?://(www\.)?tiktok\.com/.+/video/\d+")
_UNKNOWN_LANG = {"", "un", "und", "unknown", "xx", "zxx"}
_YOUTUBE_LAUNCH = pd.Timestamp("2005-04-01", tz="UTC")
_C1_CONTROLS = re.compile("[\x80-\x9f]")

CLEAN_COLUMNS = [
    "video_uid", "platform", "video_id", "url", "title", "description", "has_description",
    "creator_name", "creator_id", "creator_url", "creator_followers",
    "duration_seconds", "upload_date", "views", "likes", "comments", "shares", "saves",
    "hashtags", "tags", "category",
    "language", "language_clean", "country_code", "country_clean", "country_source",
    "transcript", "has_transcript", "transcript_status", "transcript_source",
    "transcript_language", "transcript_language_clean", "transcript_is_auto", "transcript_word_count",
    "query_ids", "query_count", "best_query_rank", "query_subtopics",
    "thumbnail_url", "retrieved_at_utc",
]


def clean_text(value: str) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    return _CTRL.sub("", value).strip()


def repair_transcript(text: str):
    """Undo the UTF-8-read-as-Latin-1 garbling found in early TikTok captions.

    Returns (text, state) where state is 'unchanged', 'repaired' or 'corrupt'.
    'corrupt' = garbled and some bytes were lost, so it cannot be recovered losslessly (text is blanked).
    """
    if not text or text.isascii():
        return text, "unchanged"
    try:
        return text.encode("latin-1").decode("utf-8"), "repaired"
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    if _C1_CONTROLS.search(text):
        return "", "corrupt"
    return text, "unchanged"


def clean_language(value: str) -> str:
    v = (value or "").strip().lower().replace("_", "-")
    primary = v.split("-")[0]
    return "unknown" if primary in _UNKNOWN_LANG else primary


def clean_country(value: str) -> str:
    v = (value or "").strip().upper()
    return v if re.fullmatch(r"[A-Z]{2}", v) else "Unknown"


def _valid_url(platform: str, url: str) -> bool:
    return bool((_YT_URL if platform == "youtube" else _TT_URL).match(url or ""))


def clean_master(dedup: pd.DataFrame):
    """Return (cleaned_df, dropped_df, report)."""
    df = dedup.copy()

    # encoding repair must run before any other text normalisation
    fixed = df["transcript"].map(repair_transcript)
    df["transcript"] = fixed.map(lambda t: t[0])
    state = fixed.map(lambda t: t[1])
    n_repaired, n_corrupt = int((state == "repaired").sum()), int((state == "corrupt").sum())
    corrupt = state == "corrupt"
    df.loc[corrupt, "transcript_status"] = "corrupt_encoding"
    df.loc[corrupt, ["transcript_source", "transcript_language", "transcript_is_auto"]] = ""

    for col in TEXT_COLS:
        df[col] = df[col].map(clean_text)
    df["hashtags"] = df["hashtags"].str.lower()

    for col in INT_COLS:
        num = pd.to_numeric(df[col], errors="coerce")
        df[col] = num.where(num >= 0).astype("Int64")
    dur = pd.to_numeric(df["duration_seconds"], errors="coerce")
    df["duration_seconds"] = dur.where(dur > 0)

    when = pd.to_datetime(df["upload_date"], errors="coerce", utc=True)
    bad_date = when.isna() | (when > pd.Timestamp.now(tz="UTC")) | (when < _YOUTUBE_LAUNCH)
    n_bad_dates = int((bad_date & (df["upload_date"].str.strip() != "")).sum())
    df["upload_date"] = when.where(~bad_date).dt.strftime("%Y-%m-%d").fillna("")

    df["video_uid"] = df["platform"] + ":" + df["video_id"]
    df["language_clean"] = df["language"].map(clean_language)
    df["transcript_language_clean"] = df["transcript_language"].map(clean_language).where(df["transcript"] != "", "")
    df["country_clean"] = df["country_code"].map(clean_country)
    # TikTok has a single caption field (stored in title), so "has description" does not apply to it
    df["has_description"] = (df["description"].str.len() > 0).astype("boolean").mask(df["platform"] == "tiktok")
    df["has_transcript"] = (df["transcript_status"] == "ok") & (df["transcript"].str.len() > 0)
    df["transcript_word_count"] = df["transcript"].str.split().str.len().fillna(0).astype("Int64")
    df["transcript_is_auto"] = df["transcript_is_auto"].map({"True": True, "False": False}).astype("boolean")

    # ---- drop rules (first matching reason wins)
    rules = [
        ("metadata_not_ok", df["metadata_status"] != "ok"),
        ("invalid_url", ~pd.Series([_valid_url(p, u) for p, u in zip(df["platform"], df["url"])], index=df.index)),
        ("invalid_duration", df["duration_seconds"].isna()),
        ("no_text_at_all", (df["title"] == "") & (df["description"] == "") & (df["transcript"] == "")),
    ]
    reason = pd.Series("", index=df.index)
    for name, mask in rules:
        reason = reason.mask((reason == "") & mask, name)

    dropped = df.loc[reason != "", ["video_uid", "platform", "video_id", "url", "metadata_status"]].copy()
    dropped["reason"] = reason[reason != ""]
    cleaned = df.loc[reason == ""].copy()
    assert cleaned["video_uid"].is_unique, "video_uid must be unique after deduplication"
    cleaned = cleaned[CLEAN_COLUMNS].reset_index(drop=True)

    by_platform = {}
    for platform, g in cleaned.groupby("platform"):
        by_platform[platform] = {
            "videos": int(len(g)),
            "has_transcript_pct": round(float(g["has_transcript"].mean() * 100), 1),
            "has_description_pct": None if g["has_description"].isna().all() else round(float(g["has_description"].dropna().mean() * 100), 1),
            "missing_upload_date": int((g["upload_date"] == "").sum()),
            "missing_views": int(g["views"].isna().sum()),
            "missing_likes": int(g["likes"].isna().sum()),
            "country_unknown_pct": round(float((g["country_clean"] == "Unknown").mean() * 100), 1),
            "language_unknown_pct": round(float((g["language_clean"] == "unknown").mean() * 100), 1),
        }
    report = {
        "rows_in": int(len(df)),
        "rows_kept": int(len(cleaned)),
        "rows_dropped": int(len(dropped)),
        "dropped_by_reason": dropped["reason"].value_counts().to_dict(),
        "invalid_or_out_of_range_dates_blanked": n_bad_dates,
        "transcripts_encoding_repaired": n_repaired,
        "transcripts_corrupt_blanked": n_corrupt,
        "by_platform": by_platform,
    }
    return cleaned, dropped, report
