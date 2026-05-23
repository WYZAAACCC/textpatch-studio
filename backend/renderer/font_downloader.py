"""Font auto-downloader for Chinese fonts.

Downloads Noto Sans CJK SC (a high-quality open-source Chinese font)
when no suitable Chinese fonts are found on the system.
"""

from __future__ import annotations
import hashlib
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Callable

import requests

logger = logging.getLogger(__name__)

FONT_DOWNLOAD_URLS = [
    "https://github.com/googlefonts/noto-cjk/releases/download/Sans2.004/03_NotoSansCJKsc.zip",
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
]

FONT_FILENAME = "NotoSansCJKsc-Regular.otf"
FONT_DISPLAY_NAME = "Noto Sans CJK SC"
FONT_EXPECTED_SIZE_MIN = 1024 * 1024  # 1 MB minimum for valid CJK font


class FontDownloadError(Exception):
    """Raised when font download fails."""


def _verify_font(path: Path) -> bool:
    """Verify that a font file is valid."""
    if not path.exists():
        return False
    if path.stat().st_size < FONT_EXPECTED_SIZE_MIN:
        return False
    suffix = path.suffix.lower()
    if suffix not in (".otf", ".ttf", ".ttc"):
        return False
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(str(path), 14)
        return font.getname()[0] != ""
    except Exception:
        return False


def _download_url(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download file from URL to destination. Returns True on success."""
    logger.info("Downloading font from %s", url)
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)

        if _verify_font(dest):
            logger.info("Font downloaded successfully: %s (%d bytes)", dest.name, downloaded)
            return True
        else:
            dest.unlink(missing_ok=True)
            logger.warning("Downloaded file is not a valid font: %s", url)
            return False

    except requests.RequestException as e:
        logger.warning("Font download failed from %s: %s", url, e)
        dest.unlink(missing_ok=True)
        return False


def download_chinese_font(
    target_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """Download a Chinese font (Noto Sans CJK SC) to the target directory.

    Args:
        target_dir: Directory to save the font. Defaults to project's fonts/ directory.
        progress_callback: Optional callback for progress updates.

    Returns:
        Path to the downloaded font file, or None if download failed.
    """
    if target_dir is None:
        target_dir = Path(__file__).parent.parent.parent.parent / "fonts"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / FONT_FILENAME

    if _verify_font(target_path):
        logger.info("Chinese font already exists at %s", target_path)
        return target_path

    def _update(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _update(f"Downloading {FONT_DISPLAY_NAME} font...")

    for url in FONT_DOWNLOAD_URLS:
        if url.endswith(".zip"):
            success, path = _download_and_extract_zip(url, target_dir, _update)
            if success:
                return path
        else:
            if _download_url(url, target_path):
                return target_path

    raise FontDownloadError(
        f"Failed to download Chinese font from all sources. "
        f"Please manually download '{FONT_DISPLAY_NAME}' and place it in: {target_dir}"
    )


def _download_and_extract_zip(
    url: str, target_dir: Path, update_callback: Callable[[str], None]
) -> tuple[bool, Optional[Path]]:
    """Download a zip file and extract the font from it."""
    update_callback(f"Downloading {FONT_DISPLAY_NAME} (zip archive)...")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Zip download failed: %s", e)
        return False, None

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=65536):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(FONT_FILENAME) or (
                    "NotoSansCJKsc" in name and name.endswith("-Regular.otf")
                ):
                    update_callback(f"Extracting {name}...")
                    zf.extract(name, target_dir)

                    if name.endswith(FONT_FILENAME):
                        result_path = target_dir / FONT_FILENAME
                    else:
                        extracted = target_dir / name
                        result_path = target_dir / FONT_FILENAME
                        if result_path.exists():
                            result_path.unlink()
                        extracted.rename(result_path)

                    if _verify_font(result_path):
                        update_callback(f"Font installed: {result_path}")
                        return True, result_path
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning("Failed to extract zip: %s", e)
    finally:
        tmp_path.unlink(missing_ok=True)

    return False, None


def ensure_chinese_font_available() -> Path:
    """Ensure at least one Chinese font is available. Downloads if needed.

    Returns:
        Path to an available Chinese font.

    Raises:
        FontDownloadError: If no font is available and download fails.
    """
    from backend.renderer.font_manager import FontManager

    fm = FontManager()

    if fm.has_any_font():
        default = fm.get_default_font_path()
        if default:
            return Path(default)

    fonts_dir = Path(__file__).parent.parent.parent.parent / "fonts"
    return download_chinese_font(target_dir=fonts_dir)
