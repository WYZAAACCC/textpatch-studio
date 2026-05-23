import re
from pathlib import Path

PROJECT_ID_RE = re.compile(r"^p_\d{8}_[a-f0-9]{8}$")
REGION_ID_RE = re.compile(r"^region_[a-f0-9]{6}$")


def validate_project_id(project_id: str) -> None:
    if not PROJECT_ID_RE.match(project_id):
        raise ValueError(f"Invalid project ID format: {project_id}")


def validate_region_id(region_id: str) -> None:
    if not REGION_ID_RE.match(region_id):
        raise ValueError(f"Invalid region ID format: {region_id}")


def ensure_child_path(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    resolved = candidate.resolve()
    if base_resolved != resolved and base_resolved not in resolved.parents:
        raise ValueError("Unsafe path: attempted path traversal")
    return resolved
