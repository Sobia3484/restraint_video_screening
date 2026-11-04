from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .utils import slugify


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    subtopic: str = ""
    language: str = "en"

    @property
    def slug(self) -> str:
        return slugify(self.text)

    @property
    def file_stem(self) -> str:
        return f"{self.id}_{self.slug}"


def load_queries(path: Path) -> list[Query]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    queries = [
        Query(
            id=r["query_id"].strip(),
            text=r["query"].strip(),
            subtopic=(r.get("subtopic") or "").strip(),
            language=(r.get("query_language") or "en").strip(),
        )
        for r in rows
        if (r.get("query") or "").strip()
    ]
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query_id values in queries file")
    texts = [q.text.lower() for q in queries]
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate query text in queries file")
    return queries
