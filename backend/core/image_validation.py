from __future__ import annotations
import warnings
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError

from backend.config import StorageConfig

ImageFile.LOAD_TRUNCATED_IMAGES = False


class InvalidImageError(ValueError):
    pass


def validate_image_file(path: Path, config: StorageConfig) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            img = Image.open(path)
            img.verify()
        except Image.DecompressionBombWarning as e:
            raise InvalidImageError("Image exceeds decompression bomb limit") from e
        except UnidentifiedImageError:
            raise InvalidImageError("File is not a valid image")
        except Exception as e:
            raise InvalidImageError(f"Invalid image file: {e}") from e

    img = Image.open(path)
    w, h = img.size

    if w <= 0 or h <= 0:
        raise InvalidImageError("Invalid image dimensions")
    if w > config.max_image_width or h > config.max_image_height:
        raise InvalidImageError(
            f"Image dimensions ({w}x{h}) exceed limit "
            f"({config.max_image_width}x{config.max_image_height})"
        )
    if w * h > config.max_image_pixels:
        raise InvalidImageError(
            f"Image pixel count ({w * h}) exceeds limit ({config.max_image_pixels})"
        )

    return img
