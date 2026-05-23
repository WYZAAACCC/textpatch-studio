"""In-memory rate limiter for external API calls (OCR/LLM)."""
from __future__ import annotations
import threading
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._project_llm_counts: dict[str, int] = defaultdict(int)
        self._project_ocr_counts: dict[str, int] = defaultdict(int)
        self._window_start = time.monotonic()

    def check_llm(self, project_id: str, max_per_project: int = 200,
                  max_per_minute: int = 60) -> None:
        with self._lock:
            now = time.monotonic()
            if now - self._window_start > 60:
                self._project_llm_counts.clear()
                self._project_ocr_counts.clear()
                self._window_start = now

            if self._project_llm_counts[project_id] >= max_per_project:
                raise RateLimitExceeded(
                    f"LLM request limit ({max_per_project}) exceeded for project {project_id}"
                )
            self._project_llm_counts[project_id] += 1

    def check_ocr(self, project_id: str, max_per_minute: int = 30) -> None:
        with self._lock:
            now = time.monotonic()
            if now - self._window_start > 60:
                self._project_llm_counts.clear()
                self._project_ocr_counts.clear()
                self._window_start = now


class RateLimitExceeded(Exception):
    pass


_limiter: RateLimit | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
