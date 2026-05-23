from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.models.region import TextRegion
from backend.models.style import TextStyle
from backend.models.render import RenderInfo
from backend.renderer.font_manager import FontManager
from backend.renderer.text_layout import layout_text
from backend.renderer.text_renderer import render_text_layer
from backend.renderer.effects import apply_stroke, apply_shadow
from backend.renderer.formula_renderer import (
    contains_formula, text_to_latex, render_formula, render_formula_unicode,
)

logger = logging.getLogger(__name__)


def render_region(
    region: TextRegion,
    image_size: tuple[int, int],
    font_manager: FontManager,
) -> tuple[Optional[Image.Image], RenderInfo]:
    if not region.final_text or not region.bbox or len(region.bbox) < 4:
        return None, RenderInfo(overflow=True)

    style = region.style
    if style is None:
        style = TextStyle()

    x1, y1, x2, y2 = region.bbox
    box_w = int(x2 - x1)
    box_h = int(y2 - y1)

    if box_w <= 0 or box_h <= 0:
        return None, RenderInfo(overflow=True)

    font_path = font_manager.get_font_path(style.font_family)
    if not font_path:
        font_path = font_manager.get_default_font_path()

    if not font_path:
        if not font_manager.has_any_font():
            logger.warning(
                "No TrueType fonts found. Install Chinese fonts (e.g. Noto Sans CJK SC) "
                "in the fonts/ directory or use system fonts."
            )
        logger.error(f"No font available for rendering region {region.id}")
        return None, RenderInfo(overflow=True)

    text = region.final_text
    overflow = False
    layer = None
    layout = None

    # Try formula rendering if text contains math notation
    if contains_formula(text):
        latex = text_to_latex(text)
        formula_img = render_formula(
            latex,
            font_size=style.font_size,
            color=style.color,
            max_width=box_w,
        )
        if formula_img is not None:
            # Scale to fit bbox while preserving aspect ratio
            lw, lh = formula_img.size
            if lw > box_w or lh > box_h:
                scale = min(box_w / lw, box_h / lh) * 0.95
                formula_img = formula_img.resize(
                    (int(lw * scale), int(lh * scale)), Image.LANCZOS
                )

            # Place on properly-sized canvas with vertical centering
            lw, lh = formula_img.size
            canvas_w = max(box_w, lw)
            canvas_h = max(box_h, lh)
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            px = (canvas_w - lw) // 2
            py = (canvas_h - lh) // 2
            canvas.paste(formula_img, (px, py), formula_img)
            layer = canvas

    if layer is None:
        # Regular text rendering path
        # Try unicode math conversion first for formula-ish text
        if contains_formula(text):
            text = render_formula_unicode(text)

        layout = layout_text(
            text=text,
            font_path=font_path,
            font_size=style.font_size,
            max_width=box_w,
            max_height=box_h,
            style=style,
        )

        if layout is None or layout.overflow:
            layout = _fit_text_to_box(
                text, font_path, box_w, box_h, style
            )

        if layout is None:
            return None, RenderInfo(overflow=True)

        overflow = layout.overflow

        if overflow and style.allow_region_expand:
            expand_w = int(box_w * 1.5)
            expand_h = int(box_h * 1.5)
            expanded_layout = _fit_text_to_box(
                text, font_path, expand_w, expand_h, style
            )
            if expanded_layout and not expanded_layout.overflow:
                layout = expanded_layout
                overflow = False

        layer = render_text_layer(
            layout, font_path, style,
            box_width=box_w, box_height=box_h,
        )

    # Stroke/shadow only for text-rendered layers (layout is populated)
    if layout is not None and style.stroke_width > 0 and style.stroke_color:
        layer = apply_stroke(layer, layout, font_path, style)

    if layout is not None and style.shadow:
        layer = apply_shadow(layer, layout, style)

    if abs(region.angle) > 0.5:
        layer = layer.rotate(-region.angle, expand=True, resample=Image.BICUBIC)

    transform = "none"
    if abs(region.angle) > 0.5:
        transform = "rotation"

    render_info = RenderInfo(
        transform=transform,
        blend_mode=style.blend_mode,
        overflow=overflow,
    )

    return layer, render_info


def _fit_text_to_box(
    text: str,
    font_path: str,
    box_w: int,
    box_h: int,
    style: TextStyle,
):
    max_size = int(style.max_font_size)
    min_size = int(style.min_font_size)

    for size in range(max_size, min_size - 1, -1):
        layout = layout_text(
            text=text,
            font_path=font_path,
            font_size=float(size),
            max_width=box_w,
            max_height=box_h,
            style=style,
        )
        if layout and not layout.overflow:
            return layout

    layout = layout_text(
        text=text,
        font_path=font_path,
        font_size=float(min_size),
        max_width=box_w,
        max_height=box_h,
        style=style,
    )
    if layout:
        layout.overflow = True
    return layout
