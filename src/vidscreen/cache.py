from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional


class JsonCache:
    """Tiny on-disk JSON cache: <root>/<namespace>/<key>.json (atomic writes)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _safe(key: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", str(key))[:150]

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{self._safe(key)}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
