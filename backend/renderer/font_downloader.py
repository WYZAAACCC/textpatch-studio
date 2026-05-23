"""Font auto-downloader for Chinese fonts.

Downloads Noto Sans CJK SC (a high-quality open-source Chinese font)
when no suitable Chinese fonts are found on the system.

Remote download is DISABLED by default for security. Set
TEXTPATCH_ENABLE_FONT_DOWNLOAD=true to enable.
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
FONT_EXPECTED_SIZE_MIN = 1024 * 1024
MAX_FONT_DOWNLOAD_BYTES = 200 * 1024 * 1024
MAX_SINGLE_FILE_BYTES = 50 * 1024 * 1024


class FontDownloadError(Exception):
    """Raised when font download fails."""


def _verify_font(path: Path) -> bool:
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


def _is_remote_download_enabled() -> bool:
    return os.environ.get("TEXTPATCH_ENABLE_FONT_DOWNLOAD", "false").lower() in (
        "1", "true", "yes"
    )


def _safe_extract_zip(zf: zipfile.ZipFile, target_dir: Path) -> Optional[Path]:
    target_dir = target_dir.resolve()
    for info in zf.infolist():
        name = info.filename
        if name.startswith("/") or ".." in Path(name).parts:
            logger.warning("Skipping unsafe zip entry: %s", name)
            continue
        if info.file_size > MAX_SINGLE_FILE_BYTES:
            logger.warning("Skipping oversized zip entry: %s (%d bytes)", name, info.file_size)
            continue
        if name.endswith(FONT_FILENAME) or (
            "NotoSansCJKsc" in name and name.endswith("-Regular.otf")
        ):
            extracted = (target_dir / Path(name).name).resolve()
            if target_dir not in extracted.parents:
                logger.warning("Skipping path traversal attempt: %s", name)
                continue
            with zf.open(info) as src:
                with open(extracted, "wb") as dst:
                    while True:
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        dst.write(chunk)
            if _verify_font(extracted):
                return extracted
    return None


def _download_url(url: str, dest: Path, timeout: int = 120) -> bool:
    logger.info("Downloading font from %s", url)
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        total = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)
                if total > MAX_FONT_DOWNLOAD_BYTES:
                    raise FontDownloadError("Font download too large, exceeding 200 MB limit")

        if _verify_font(dest):
            logger.info("Font downloaded successfully (%d bytes)", total)
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

    Remote download must be enabled via TEXTPATCH_ENABLE_FONT_DOWNLOAD=true.
    """
    if not _is_remote_download_enabled():
        raise FontDownloadError(
            "Remote font download is disabled by default for security. "
            "Set TEXTPATCH_ENABLE_FONT_DOWNLOAD=true to enable, or place a font "
            "file manually in the fonts/ directory."
        )

    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent.parent / "fonts"

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
    update_callback(f"Downloading {FONT_DISPLAY_NAME} (zip archive)...")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Zip download failed: %s", e)
        return False, None

    total = 0
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=65536):
            tmp.write(chunk)
            total += len(chunk)
            if total > MAX_FONT_DOWNLOAD_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise FontDownloadError("Font zip download too large, exceeding 200 MB limit")
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            result_path = _safe_extract_zip(zf, target_dir)
            if result_path:
                update_callback(f"Font installed: {result_path}")
                return True, result_path
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning("Failed to extract zip: %s", e)
    finally:
        tmp_path.unlink(missing_ok=True)

    return False, None


def ensure_chinese_font_available() -> Path:
    from backend.renderer.font_manager import FontManager

    fm = FontManager()

    if fm.has_any_font():
        default = fm.get_default_font_path()
        if default:
            return Path(default)

    fonts_dir = Path(__file__).resolve().parent.parent / "fonts"
    return download_chinese_font(target_dir=fonts_dir)
