"""Centralized dependency injection for FastAPI routes.

Provides singleton Pipeline and cached engine/font-manager lookups
to avoid repeated initialization across requests.
"""
from __future__ import annotations

from backend.config import app_config
from backend.core.pipeline import Pipeline

_pipeline_singleton: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = Pipeline(app_config)
    return _pipeline_singleton


def reset_pipeline():
    """Reset the singleton (useful for tests)."""
    global _pipeline_singleton
    _pipeline_singleton = None
