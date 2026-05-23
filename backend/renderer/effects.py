from __future__ import annotations
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from backend.renderer.text_layout import TextLayout
from backend.models.style import TextStyle


def apply_stroke(
    layer: Image.Image,
    layout: TextLayout,
    font_path: str,
    style: TextStyle,
) -> Image.Image:
    if not style.stroke_color or style.stroke_width <= 0:
        return layer

    stroke_w = max(1, int(style.stroke_width))
    stroke_color = style.stroke_color
    if len(stroke_color) == 3:
        stroke_color = (*stroke_color, 255)

    canvas_pad = stroke_w * 2
    canvas_w = layer.width + canvas_pad * 2
    canvas_h = layer.height + canvas_pad * 2

    stroke_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke_layer)

    try:
        font = ImageFont.truetype(font_path, int(layout.font_size))
    except Exception:
        font = ImageFont.load_default()

    for text_line in layout.lines:
        if not text_line.text:
            continue

        x = text_line.x + canvas_pad
        y = text_line.y + canvas_pad

        for dx in range(-stroke_w, stroke_w + 1):
            for dy in range(-stroke_w, stroke_w + 1):
                if dx * dx + dy * dy <= stroke_w * stroke_w:
                    draw.text(
                        (x + dx, y + dy),
                        text_line.text,
                        font=font,
                        fill=stroke_color,
                    )

    text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)

    for text_line in layout.lines:
        if not text_line.text:
            continue

        x = text_line.x + canvas_pad
        y = text_line.y + canvas_pad

        color = style.color
        if len(color) == 3:
            color = (*color, 255)

        text_draw.text((x, y), text_line.text, font=font, fill=color)

    result = Image.alpha_composite(stroke_layer, text_layer)

    cropped = result.crop((
        canvas_pad,
        canvas_pad,
        canvas_pad + layer.width,
        canvas_pad + layer.height,
    ))

    return cropped


def apply_shadow(
    layer: Image.Image,
    layout: TextLayout,
    style: TextStyle,
) -> Image.Image:
    if not style.shadow:
        return layer

    shadow = style.shadow
    dx = shadow.dx
    dy = shadow.dy
    blur = max(1, int(shadow.blur))
    shadow_color = shadow.color
    if len(shadow_color) == 3:
        shadow_color = (*shadow_color, 128)

    shadow_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))

    alpha = layer.split()[3]
    shadow_alpha = alpha.point(lambda p: int(p * shadow_color[3] / 255) if len(shadow_color) > 3 else p)

    shadow_colored = Image.new("RGBA", layer.size, (*shadow_color[:3], 0))
    shadow_colored.putalpha(shadow_alpha)

    if blur > 1:
        from PIL import ImageFilter
        shadow_colored = shadow_colored.filter(ImageFilter.GaussianBlur(radius=blur))

    canvas = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    canvas.paste(shadow_colored, (int(dx), int(dy)), shadow_colored)
    canvas = Image.alpha_composite(canvas, layer)

    return canvas
