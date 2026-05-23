import pytest
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from backend.models.region import TextRegion
from backend.models.style import TextStyle
from backend.models.render import RenderInfo
from backend.renderer.font_manager import FontManager
from backend.renderer.text_layout import layout_text
from backend.renderer.text_renderer import render_text_layer
from backend.core.rendering import render_region, _fit_text_to_box
from backend.core.compositing import composite_layers


class TestTextLayout:
    def test_layout_simple_text(self):
        fm = FontManager()
        font_path = fm.get_default_font_path()
        if not font_path:
            pytest.skip("No font available")

        layout = layout_text("Hello", font_path, 24, 200, 50)
        assert layout is not None
        assert layout.width > 0
        assert layout.height > 0
        assert not layout.overflow

    def test_layout_overflow(self):
        fm = FontManager()
        font_path = fm.get_default_font_path()
        if not font_path:
            pytest.skip("No font available")

        layout = layout_text("Very long text that should overflow", font_path, 48, 50, 20)
        assert layout is not None
        assert layout.overflow is True

    def test_layout_empty_text(self):
        fm = FontManager()
        font_path = fm.get_default_font_path()
        if not font_path:
            pytest.skip("No font available")

        layout = layout_text("", font_path, 24, 200, 50)
        assert layout is None

    def test_layout_multiline(self):
        fm = FontManager()
        font_path = fm.get_default_font_path()
        if not font_path:
            pytest.skip("No font available")

        layout = layout_text("Line1\nLine2", font_path, 24, 200, 200)
        assert layout is not None
        assert len(layout.lines) == 2


class TestTextRenderer:
    def test_render_text_layer(self):
        fm = FontManager()
        font_path = fm.get_default_font_path()
        if not font_path:
            pytest.skip("No font available")

        layout = layout_text("Test", font_path, 24, 200, 50)
        assert layout is not None

        style = TextStyle(font_size=24)
        layer = render_text_layer(layout, font_path, style)
        assert layer is not None
        assert layer.size[0] > 0
        assert layer.size[1] > 0
        assert layer.mode == "RGBA"


class TestFontManager:
    def test_list_fonts(self):
        fm = FontManager()
        fonts = fm.list_available_fonts()
        assert isinstance(fonts, list)

    def test_get_default_font(self):
        fm = FontManager()
        path = fm.get_default_font_path()
        assert path is None or isinstance(path, str)


class TestRenderRegion:
    def test_render_region_no_text(self):
        region = TextRegion.create(
            polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
            bbox=[0, 0, 100, 50],
        )
        region.final_text = ""
        fm = FontManager()
        layer, info = render_region(region, (800, 600), fm)
        assert layer is None
        assert info.overflow is True

    def test_render_region_with_text(self):
        fm = FontManager()
        font_path = fm.get_default_font_path()
        if not font_path:
            pytest.skip("No font available")

        region = TextRegion.create(
            polygon=[[10, 10], [300, 10], [300, 60], [10, 60]],
            bbox=[10, 10, 300, 60],
        )
        region.final_text = "测试文字"
        region.style = TextStyle(font_size=24)

        layer, info = render_region(region, (800, 600), fm)
        assert layer is not None or info.overflow


class TestFitTextToBox:
    def test_fit_text_to_box(self):
        fm = FontManager()
        font_path = fm.get_default_font_path()
        if not font_path:
            pytest.skip("No font available")

        style = TextStyle(min_font_size=8, max_font_size=48)
        layout = _fit_text_to_box("Test", font_path, 200, 50, style)
        assert layout is not None

    def test_fit_text_too_small(self):
        fm = FontManager()
        font_path = fm.get_default_font_path()
        if not font_path:
            pytest.skip("No font available")

        style = TextStyle(min_font_size=8, max_font_size=48)
        layout = _fit_text_to_box("Very long text", font_path, 10, 5, style)
        if layout:
            assert layout.overflow is True


class TestComposite:
    def test_composite_no_layers(self):
        base = Image.new("RGBA", (800, 600), (255, 255, 255, 255))
        result = composite_layers(base, [], {})
        assert result.size == (800, 600)

    def test_composite_with_layer(self):
        base = Image.new("RGBA", (800, 600), (255, 255, 255, 255))
        region = TextRegion.create(
            polygon=[[10, 10], [200, 10], [200, 50], [10, 50]],
            bbox=[10, 10, 200, 50],
        )
        region.final_text = "Test"
        region.status = "approved"
        region.render = RenderInfo()

        layer = Image.new("RGBA", (190, 40), (255, 0, 0, 200))
        layers = {region.id: layer}

        result = composite_layers(base, [region], layers)
        assert result.size == (800, 600)
