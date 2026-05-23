from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from PIL import Image

from backend.models.project import Project
from backend.storage.project_store import ProjectStore
from backend.storage.file_store import FileStore

logger = logging.getLogger(__name__)


def export_project(
    project: Project,
    project_store: ProjectStore,
    file_store: FileStore,
    format: str = "png",
    quality: int = 100,
    include_project_file: bool = True,
) -> Path:
    project_dir = project_store._project_dir(project.id)

    final_path = project_dir / "final.png"
    if not final_path.exists():
        clean_base_path = project_dir / "clean_base.png"
        original_path = project_dir / "original.png"

        if clean_base_path.exists():
            src = clean_base_path
        elif original_path.exists():
            src = original_path
        else:
            raise FileNotFoundError(f"No base image found for project {project.id}")

        shutil.copy2(src, final_path)

    if include_project_file:
        project_store.save(project)

    if format == "jpg" or format == "jpeg":
        img = Image.open(final_path)
        jpg_path = project_dir / "final.jpg"
        img.save(str(jpg_path), "JPEG", quality=quality)
        return jpg_path

    return final_path
