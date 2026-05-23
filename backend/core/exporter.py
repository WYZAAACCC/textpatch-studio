from __future__ import annotations
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from PIL import Image

from backend.models.project import Project
from backend.storage.project_store import ProjectStore
from backend.storage.file_store import FileStore

logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "zip": "application/zip",
}


def export_project(
    project: Project,
    project_store: ProjectStore,
    file_store: FileStore,
    format: str = "png",
    quality: int = 95,
    scale: float = 1.0,
    include_project_file: bool = True,
) -> Path:
    project_dir = project_store._project_dir(project.id, create=True)

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

    img = Image.open(final_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    if format == "zip":
        zip_path = project_dir / "project_export.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if final_path.exists():
                zf.write(final_path, "final.png")
            orig = project_dir / "original.png"
            if orig.exists():
                zf.write(orig, "original.png")
            clean = project_dir / "clean_base.png"
            if clean.exists():
                zf.write(clean, "clean_base.png")
            regions_dir = project_dir / "regions"
            if regions_dir.exists():
                for f in regions_dir.iterdir():
                    if f.is_file():
                        zf.write(f, f"regions/{f.name}")
            if include_project_file:
                with open(project_dir / "project.json", "r", encoding="utf-8") as f:
                    zf.writestr("project.json", f.read())
        return zip_path

    if format in ("jpg", "jpeg"):
        out_path = project_dir / f"final_{scale}x.jpg"
        img.save(str(out_path), "JPEG", quality=quality)
    elif format == "webp":
        out_path = project_dir / f"final_{scale}x.webp"
        img.save(str(out_path), "WEBP", quality=quality)
    else:
        out_path = project_dir / f"final_{scale}x.png"
        img.save(str(out_path), "PNG")

    return out_path
