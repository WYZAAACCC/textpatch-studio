"""Tests for upload security: image validation, chunked upload."""
import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from PIL import Image

from backend.core.image_validation import validate_image_file, InvalidImageError
from backend.config import StorageConfig


class TestImageValidation:
    def make_config(self, **kw):
        defaults = {
            "max_image_width": 12000,
            "max_image_height": 12000,
            "max_image_pixels": 40_000_000,
        }
        defaults.update(kw)
        return StorageConfig(**defaults)

    def test_valid_png(self, tmp_path):
        img_path = tmp_path / "valid.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)
        result = validate_image_file(img_path, self.make_config())
        assert result.size == (100, 100)

    def test_invalid_file_not_image(self, tmp_path):
        path = tmp_path / "not_image.png"
        path.write_text("not an image")
        with pytest.raises(InvalidImageError):
            validate_image_file(path, self.make_config())

    def test_oversized_dimensions(self, tmp_path):
        img_path = tmp_path / "large.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)
        with pytest.raises(InvalidImageError):
            validate_image_file(img_path, self.make_config(max_image_width=50))

    def test_too_many_pixels(self, tmp_path):
        img_path = tmp_path / "big.png"
        img = Image.new("RGB", (2000, 2000), color="red")
        img.save(img_path)
        with pytest.raises(InvalidImageError):
            validate_image_file(img_path, self.make_config(max_image_pixels=1_000_000))
