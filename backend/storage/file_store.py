from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional

from PIL import Image

from backend.storage.path_safety import validate_project_id


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}


class FileStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str, create: bool = False) -> Path:
        from backend.storage.project_store import ProjectStore
        store = ProjectStore(self.data_dir)
        return store._project_dir(project_id, create=create)

    def save_original(self, project_id: str, source_path: Path, filename: str) -> Path:
        from backend.config import app_config
        from backend.core.image_validation import validate_image_file, InvalidImageError

        validate_project_id(project_id)
        img = validate_image_file(source_path, app_config.storage)

        project_dir = self._project_dir(project_id, create=True)
        original_path = project_dir / "original.png"

        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img_rgb = bg
        else:
            img_rgb = img.convert("RGB") if img.mode != "RGB" else img
        img_rgb.save(str(original_path), "PNG")

        preview_path = project_dir / "preview.jpg"
        preview = img_rgb.copy()
        preview.thumbnail((1200, 1200))
        preview.save(str(preview_path), "JPEG", quality=85)

        return original_path

    def save_clean_base(self, project_id: str, image: Image.Image) -> Path:
        project_dir = self._project_dir(project_id, create=True)
        path = project_dir / "clean_base.png"
        image.save(str(path), "PNG")
        return path

    def save_final(self, project_id: str, image: Image.Image) -> Path:
        project_dir = self._project_dir(project_id, create=True)
        path = project_dir / "final.png"
        image.save(str(path), "PNG")
        return path

    def get_original(self, project_id: str) -> Optional[Path]:
        path = self._project_dir(project_id) / "original.png"
        return path if path.exists() else None

    def get_clean_base(self, project_id: str) -> Optional[Path]:
        path = self._project_dir(project_id) / "clean_base.png"
        return path if path.exists() else None

    def get_final(self, project_id: str) -> Optional[Path]:
        path = self._project_dir(project_id) / "final.png"
        return path if path.exists() else None

    def get_preview(self, project_id: str) -> Optional[Path]:
        path = self._project_dir(project_id) / "preview.jpg"
        return path if path.exists() else None

    def save_region_image(
        self, project_id: str, region_id: str, suffix: str, image: Image.Image
    ) -> Path:
        from backend.storage.project_store import ProjectStore
        store = ProjectStore(self.data_dir)
        store._regions_dir(project_id, create=True)
        path = store.get_region_path(project_id, region_id, suffix)
        image.save(str(path), "PNG")
        return path

    def get_region_image(
        self, project_id: str, region_id: str, suffix: str
    ) -> Optional[Path]:
        from backend.storage.project_store import ProjectStore
        store = ProjectStore(self.data_dir)
        path = store.get_region_path(project_id, region_id, suffix)
        return path if path.exists() else None

    @staticmethod
    def is_allowed_file(filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_EXTENSIONS
