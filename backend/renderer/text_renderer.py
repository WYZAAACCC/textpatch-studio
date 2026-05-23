from __future__ import annotations
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from backend.renderer.text_layout import TextLayout
from backend.models.style import TextStyle


def render_text_layer(
    layout: TextLayout,
    font_path: str,
    style: TextStyle,
    box_width: int = 0,
    box_height: int = 0,
) -> Image.Image:
    canvas_w = max(int(box_width) if box_width > 0 else int(layout.width) + 20, 1)
    canvas_h = max(int(box_height) if box_height > 0 else int(layout.height) + 20, 1)

    layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    try:
        font = ImageFont.truetype(font_path, int(layout.font_size))
    except Exception:
        font = ImageFont.load_default()

    color = style.color
    if len(color) == 3:
        color = (*color, 255)

    bounding_height = layout.height

    v_align = getattr(style, "v_align", "center")
    v_offset_ratio = getattr(style, "v_offset_ratio", -1.0)

    if v_offset_ratio >= 0:
        # Precise positioning: use detected text origin within the bbox
        available_h = max(canvas_h - bounding_height, 1)
        y_offset = v_offset_ratio * available_h
        y_offset = max(0, min(float(canvas_h - bounding_height), y_offset))
    elif v_align == "top":
        y_offset = 2
    elif v_align == "bottom":
        y_offset = max(0, canvas_h - bounding_height - 2)
    else:
        y_offset = max(0, (canvas_h - bounding_height) / 2)

    for text_line in layout.lines:
        if not text_line.text:
            continue

        x = text_line.x
        y = text_line.y + y_offset

        if style.letter_spacing > 0:
            _draw_with_spacing(draw, text_line.text, x, y, font, color, style.letter_spacing)
        else:
            draw.text((x, y), text_line.text, font=font, fill=color)

    return layer


def _draw_with_spacing(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: float,
    y: float,
    font: ImageFont.FreeTypeFont,
    color: tuple,
    spacing: float,
):
    current_x = x
    for char in text:
        draw.text((current_x, y), char, font=font, fill=color)
        try:
            bbox = font.getbbox(char)
            char_w = bbox[2] - bbox[0]
        except Exception:
            char_w = 14
        current_x += char_w + spacing
