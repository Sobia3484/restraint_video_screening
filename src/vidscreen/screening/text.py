from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=None)
def _fold_char(ch: str) -> str:
    """Strip diacritics from Latin letters only (so Cyrillic/Arabic/Devanagari stay intact)."""
    if ch.isascii():
        return ch
    if "LATIN" in unicodedata.name(ch, ""):
        base = unicodedata.normalize("NFKD", ch)
        return "".join(c for c in base if not unicodedata.combining(c))
    return ch


def fold(text: str) -> str:
    return "".join(_fold_char(c) for c in unicodedata.normalize("NFC", (text or "").lower()))


def clean_field(value) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()


def build_text(row, description_chars: int = 1500) -> str:
    """The single text the models see: title + description + hashtags + tags (no transcript, by design)."""
    hashtags = " ".join("#" + h for h in clean_field(row["hashtags"]).split("|") if h)
    tags = " ".join(clean_field(row["tags"]).split("|"))
    parts = [clean_field(row["title"]), clean_field(row["description"])[:description_chars], hashtags, tags]
    return " ".join(p for p in parts if p)