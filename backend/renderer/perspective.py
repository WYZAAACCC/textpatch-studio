from __future__ import annotations
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.renderer.text_layout import TextLayout
from backend.models.style import TextStyle


def render_perspective_text(
    base_image: Image.Image,
    text: str,
    target_quad: list,
    style: TextStyle,
    font_path: str,
) -> Image.Image:
    rect_w, rect_h = _estimate_rect_size(target_quad)

    layer = Image.new("RGBA", (int(rect_w), int(rect_h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    try:
        font = ImageFont.truetype(font_path, int(style.font_size))
    except Exception:
        font = ImageFont.load_default()

    color = style.color
    if len(color) == 3:
        color = (*color, 255)

    draw.text((0, 0), text, font=font, fill=color)

    src_quad = np.float32([
        [0, 0],
        [rect_w, 0],
        [rect_w, rect_h],
        [0, rect_h],
    ])
    dst_quad = np.float32(target_quad)

    H = cv2.getPerspectiveTransform(src_quad, dst_quad)

    layer_np = np.array(layer)
    warped = cv2.warpPerspective(
        layer_np,
        H,
        (base_image.width, base_image.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    warped_pil = Image.fromarray(warped)
    result = Image.alpha_composite(base_image.convert("RGBA"), warped_pil)
    return result.convert("RGB")


def _estimate_rect_size(quad: list) -> tuple[float, float]:
    if len(quad) < 4:
        return (100, 50)

    w1 = ((quad[1][0] - quad[0][0]) ** 2 + (quad[1][1] - quad[0][1]) ** 2) ** 0.5
    w2 = ((quad[2][0] - quad[3][0]) ** 2 + (quad[2][1] - quad[3][1]) ** 2) ** 0.5
    h1 = ((quad[3][0] - quad[0][0]) ** 2 + (quad[3][1] - quad[0][1]) ** 2) ** 0.5
    h2 = ((quad[2][0] - quad[1][0]) ** 2 + (quad[2][1] - quad[1][1]) ** 2) ** 0.5

    return (max(w1, w2), max(h1, h2))
