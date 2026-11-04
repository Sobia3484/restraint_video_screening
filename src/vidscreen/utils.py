from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

_HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
log = logging.getLogger("vidscreen")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:max_len].strip("_") or "query"


def to_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def unix_to_date(ts) -> str:
    n = to_int(ts)
    if n is None:
        return ""
    return datetime.fromtimestamp(n, tz=timezone.utc).strftime("%Y-%m-%d")


def yyyymmdd_to_iso(value) -> str:
    if value and len(str(value)) == 8 and str(value).isdigit():
        v = str(value)
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"
    return ""


def extract_hashtags(*texts: Optional[str]) -> list[str]:
    seen, out = set(), []
    for text in texts:
        for tag in _HASHTAG_RE.findall(text or ""):
            t = tag.lower()
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def join_pipe(items: Iterable) -> str:
    return "|".join(str(i) for i in items if i not in (None, ""))


def word_count(text: Optional[str]) -> int:
    return len(text.split()) if text else 0


def polite_sleep(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(seconds * random.uniform(0.5, 1.5))


def retry(
    fn: Callable,
    attempts: int = 3,
    base_delay: float = 2.0,
    retry_on: tuple = (Exception,),
    should_retry: Optional[Callable[[Exception], bool]] = None,
):
    """Call fn(); retry with exponential backoff + jitter on `retry_on` errors."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:  # type: ignore[misc]
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            log.debug("retry %d/%d after %s: %s (sleep %.1fs)", attempt, attempts, type(exc).__name__, exc, delay)
            time.sleep(delay)
