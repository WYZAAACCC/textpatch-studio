from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from PIL import ImageFont

from backend.models.style import TextStyle


@dataclass
class TextLine:
    text: str
    x: float
    y: float
    width: float
    height: float


@dataclass
class TextLayout:
    text: str
    lines: list[TextLine]
    width: float
    height: float
    font_size: float
    overflow: bool = False


def layout_text(
    text: str,
    font_path: str,
    font_size: float,
    max_width: int,
    max_height: int,
    style: Optional[TextStyle] = None,
) -> Optional[TextLayout]:
    if not text or font_size <= 0:
        return None

    try:
        font = ImageFont.truetype(font_path, int(font_size))
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            return None

    if style is None:
        style = TextStyle()

    line_spacing = font_size * (style.line_height - 1.0)
    letter_spacing = style.letter_spacing

    raw_lines = text.split("\n")
    wrapped_lines = []

    for raw_line in raw_lines:
        if not raw_line:
            wrapped_lines.append("")
            continue

        if style.allow_wrap:
            line_wrapped = _wrap_line(raw_line, font, max_width, letter_spacing)
            wrapped_lines.extend(line_wrapped)
        else:
            wrapped_lines.append(raw_line)

    ascent, descent = font.getmetrics()
    line_h = ascent + descent

    text_lines = []
    current_y = 0.0

    for line_text in wrapped_lines:
        if not line_text:
            line_w = 0
        else:
            bbox = font.getbbox(line_text)
            line_w = bbox[2] - bbox[0]
            line_w += len(line_text) * letter_spacing

        text_line = TextLine(
            text=line_text,
            x=0,
            y=current_y,
            width=line_w,
            height=line_h,
        )
        text_lines.append(text_line)
        current_y += line_h + line_spacing

    total_width = max((tl.width for tl in text_lines), default=0)
    total_height = current_y - line_spacing if text_lines else 0

    if style.align == "center":
        for tl in text_lines:
            tl.x = (max_width - tl.width) / 2
    elif style.align == "right":
        for tl in text_lines:
            tl.x = max_width - tl.width

    overflow = total_width > max_width or total_height > max_height

    return TextLayout(
        text=text,
        lines=text_lines,
        width=total_width,
        height=total_height,
        font_size=font_size,
        overflow=overflow,
    )


def _wrap_line(text: str, font, max_width: int, letter_spacing: float = 0) -> list[str]:
    if not text:
        return [""]

    result = []
    current = ""

    for char in text:
        test = current + char
        try:
            bbox = font.getbbox(test)
            w = bbox[2] - bbox[0] + len(test) * letter_spacing
        except Exception:
            w = len(test) * 14

        if w > max_width and current:
            result.append(current)
            current = char
        else:
            current = test

    if current:
        result.append(current)

    return result if result else [""]
