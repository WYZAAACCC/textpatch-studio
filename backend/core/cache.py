"""Disk-based cache for OCR and LLM results to avoid redundant API calls."""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = "cache"


class StageCache:
    def __init__(self, project_dir: Path):
        self.cache_dir = project_dir / CACHE_DIR_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _path(self, prefix: str, key: str) -> Path:
        return self.cache_dir / f"{prefix}_{self._hash(key)}.json"

    def get(self, prefix: str, key: str) -> Optional[dict]:
        path = self._path(prefix, key)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                path.unlink(missing_ok=True)
        return None

    def set(self, prefix: str, key: str, data: dict) -> None:
        path = self._path(prefix, key)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(path)


def make_ocr_key(image_path: Path, engine: str, region_count: int) -> str:
    stat = image_path.stat()
    return f"{image_path.name}_{stat.st_size}_{engine}_{region_count}"


def make_llm_key(ocr_text: str, model: str, risk_flags: list[str]) -> str:
    flags = ",".join(sorted(risk_flags)) if risk_flags else "none"
    return f"{ocr_text}_{model}_{flags}"
