from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFont

from backend.models.region import TextRegion
from backend.models.style import TextStyle, ShadowStyle
from backend.renderer.font_manager import FontManager

logger = logging.getLogger(__name__)

_font_manager = None


def _get_font_manager() -> FontManager:
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager


def infer_style(image: np.ndarray, region: TextRegion) -> TextStyle:
    if not region.bbox or len(region.bbox) < 4:
        return TextStyle()

    x1, y1, x2, y2 = [int(v) for v in region.bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return TextStyle()

    crop = image[y1:y2, x1:x2]
    region_h = y2 - y1
    region_w = x2 - x1

    font_size = _infer_font_size(region_w, region_h, region.final_text)
    color = _infer_color(crop)
    is_bold = _infer_bold(crop)

    style = TextStyle(
        font_size=font_size,
        color=color,
        min_font_size=max(6, font_size * 0.3),
        max_font_size=min(200, font_size * 2.0),
        allow_region_expand=True,
    )

    if is_bold:
        style.font_weight = "bold"

    stroke_info = _detect_stroke(crop)
    if stroke_info:
        style.stroke_width = stroke_info["width"]
        style.stroke_color = stroke_info["color"]

    shadow_info = _detect_shadow(crop)
    if shadow_info:
        style.shadow = ShadowStyle(
            dx=shadow_info["dx"],
            dy=shadow_info["dy"],
            blur=shadow_info["blur"],
            color=shadow_info["color"],
        )

    align = _detect_alignment(region_w, region_h, crop)
    if align:
        style.align = align

    v_align_info = _detect_v_align(crop)
    if v_align_info:
        style.v_align = v_align_info["align"]
        style.v_offset_ratio = v_align_info["offset_ratio"]

    return style


def _infer_font_size(region_w: int, region_h: int, text: str) -> float:
    if not text:
        return float(region_h * 0.8)

    fm = _get_font_manager()
    font_path = fm.get_default_font_path()

    lines = text.split("\n")
    line_count = len(lines)
    available_h = region_h / line_count
    size_by_height = available_h * 0.85

    # Cap: for sparse/short text in a large bbox, don't let font
    # exceed what would fill 90% of the width with the text.
    if region_w > 0:
        char_count = len(max(lines, key=len))
        max_reasonable = (region_w / max(char_count, 1)) * 0.9
        size_by_height = min(size_by_height, max_reasonable)

    if font_path:
        try:
            longest_line = max(lines, key=len)
            char_count = len(longest_line)

            if char_count > 0:
                test_size = int(size_by_height)
                if test_size < 8:
                    test_size = 8
                font = ImageFont.truetype(font_path, test_size)
                bbox = font.getbbox(longest_line)
                text_w = bbox[2] - bbox[0]

                if text_w > 0:
                    # Only apply wrapping adjustment for extremely long text
                    if text_w > region_w * 2 and region_w > 0:
                        est_lines = max(line_count, int(text_w / region_w) + 1)
                        available_h_wrapped = region_h / est_lines
                        size_by_height = available_h_wrapped * 0.85

                    scale = region_w / text_w
                    size_by_width = test_size * scale * 0.9

                    # Only apply height floor for genuinely long text (prevents
                    # tiny fonts when many chars need to fit). For short text,
                    # let width constraint properly shrink the size.
                    if char_count > 15:
                        size_by_width = max(size_by_width, size_by_height * 0.2)

                    return max(min(size_by_height, size_by_width), 6.0)
        except Exception:
            pass

    return float(size_by_height)


def _infer_color(crop: np.ndarray) -> tuple:
    """Infer text color from the solid text core, excluding anti-aliased edges.

    AI-generated text can have 90%+ AA pixels contaminating the median.
    We erode the mask to isolate solid text pixels for a cleaner sample.
    """
    if len(crop.shape) == 2:
        gray = crop.copy()
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_mask = binary == 0
    bright_mask = binary == 255

    dark_count = np.sum(dark_mask)
    bright_count = np.sum(bright_mask)

    text_mask = dark_mask if dark_count <= bright_count else bright_mask
    is_dark_text = dark_count <= bright_count

    if not np.any(text_mask):
        return (0, 0, 0, 255)

    # Erode to exclude anti-aliased edge pixels, keeping only solid core
    h, w = text_mask.shape
    erode_size = 1
    if min(h, w) > 60:
        erode_size = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core_mask = cv2.erode(
        text_mask.astype(np.uint8), kernel, iterations=erode_size
    ).astype(bool)

    core_count = np.sum(core_mask)

    # Use core pixels if enough remain, otherwise fall back to percentiles
    if len(crop.shape) == 2:
        if core_count >= 20:
            color_val = int(np.median(gray[core_mask]))
        else:
            pct = 25 if is_dark_text else 75
            color_val = int(np.percentile(gray[text_mask], pct))
        return (color_val, color_val, color_val, 255)

    if core_count >= 20:
        b = int(np.median(crop[:, :, 0][core_mask]))
        g = int(np.median(crop[:, :, 1][core_mask]))
        r = int(np.median(crop[:, :, 2][core_mask]))
    else:
        pct = 25 if is_dark_text else 75
        b = int(np.percentile(crop[:, :, 0][text_mask], pct))
        g = int(np.percentile(crop[:, :, 1][text_mask], pct))
        r = int(np.percentile(crop[:, :, 2][text_mask], pct))

    return (r, g, b, 255)


def _infer_bold(crop: np.ndarray) -> bool:
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = np.sum(binary == 0) / (binary.shape[0] * binary.shape[1])

    return dark_ratio > 0.35


def _detect_stroke(crop: np.ndarray) -> Optional[dict]:
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape
    if h < 10 or w < 10:
        return None

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_count = np.sum(binary == 0)
    bright_count = np.sum(binary == 255)
    text_mask = binary == 0 if dark_count <= bright_count else binary == 255

    if not np.any(text_mask):
        return None

    # Use a 2-pixel dilation to detect potential stroke
    dilated = cv2.dilate(
        text_mask.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    stroke_region = (dilated > 0) & ~text_mask

    if not np.any(stroke_region):
        return None

    stroke_pixels = crop[stroke_region]
    text_pixels_full = crop[text_mask]
    if len(stroke_pixels) == 0 or len(text_pixels_full) == 0:
        return None

    if len(stroke_pixels.shape) == 1:
        stroke_median = int(np.median(stroke_pixels))
        stroke_color = (stroke_median, stroke_median, stroke_median, 255)
    else:
        b = int(np.median(stroke_pixels[:, 0]))
        g = int(np.median(stroke_pixels[:, 1]))
        r = int(np.median(stroke_pixels[:, 2]))
        stroke_color = (r, g, b, 255)

    # Stroke should occupy a significant portion relative to text
    stroke_ratio = np.sum(stroke_region) / (np.sum(text_mask) + 1)
    if stroke_ratio < 0.25:
        return None

    stroke_width = min(3.0, max(1.0, stroke_ratio * 2))

    if len(crop.shape) == 3:
        text_pixels_color = text_pixels_full.astype(np.float64)
        stroke_pixels_color = stroke_pixels.astype(np.float64)

        text_mean_color = text_pixels_color.mean(axis=0)
        stroke_mean_color = stroke_pixels_color.mean(axis=0)
        color_diff = np.mean(np.abs(text_mean_color - stroke_mean_color))

        # Stroke must be clearly different from text color
        if color_diff < 50:
            return None

        # Get background color from image border
        bg_border = _get_bg_border_color(crop, text_mask, dilated)
        if bg_border is not None:
            bg_border_arr = np.array(bg_border, dtype=np.float64)
            stroke_bg_diff = np.mean(np.abs(stroke_mean_color - bg_border_arr))

            # Stroke must be distinguishable from background
            if stroke_bg_diff < 35:
                return None

            # Stroke should be between text and background in color
            text_bg_diff = np.mean(np.abs(text_mean_color - bg_border_arr))
            if stroke_bg_diff > text_bg_diff * 1.2 and color_diff < 70:
                # Stroke is closer to background than text is — likely anti-aliasing
                return None

    return {
        "width": stroke_width,
        "color": stroke_color,
    }


def _get_bg_border_color(crop: np.ndarray, text_mask: np.ndarray, dilated: np.ndarray) -> Optional[np.ndarray]:
    h, w = crop.shape[:2]
    border_region = np.zeros((h, w), dtype=bool)
    border_size = 2
    border_region[:border_size, :] = True
    border_region[-border_size:, :] = True
    border_region[:, :border_size] = True
    border_region[:, -border_size:] = True

    bg_border = border_region & ~text_mask & ~(dilated > 0)
    if not np.any(bg_border):
        return None

    if len(crop.shape) == 3:
        return crop[bg_border].mean(axis=0)
    else:
        return np.array([float(crop[bg_border].mean())])


def _detect_shadow(crop: np.ndarray) -> Optional[dict]:
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape
    # Require larger region for reliable shadow detection
    if h < 20 or w < 20:
        return None

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_count = np.sum(binary == 0)
    bright_count = np.sum(binary == 255)
    all_dark_mask = binary == 0 if dark_count <= bright_count else binary == 255

    if not np.any(all_dark_mask):
        return None

    dark_pixels = gray[all_dark_mask]
    if len(dark_pixels) < 20:
        return None

    dark_median = np.median(dark_pixels)
    dark_std = np.std(dark_pixels)

    # Core text: darkest pixels
    core_threshold = dark_median + dark_std * 0.3
    core_text_mask = all_dark_mask & (gray <= core_threshold)

    if np.sum(core_text_mask) < 15:
        core_text_mask = all_dark_mask & (gray <= dark_median)

    if np.sum(core_text_mask) < 10:
        return None

    shadow_mask = all_dark_mask & ~core_text_mask

    # Need enough shadow pixels to be convincing
    shadow_count = np.sum(shadow_mask)
    core_count = np.sum(core_text_mask)
    if shadow_count < 30 or shadow_count < core_count * 0.25:
        return None

    core_uint8 = core_text_mask.astype(np.uint8) * 255

    shadow_candidates = []

    for dx, dy in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (5, 5), (4, 5), (5, 4)]:
        shifted = np.zeros_like(core_uint8)
        src_y1 = max(0, -dy)
        src_y2 = min(h, h - dy)
        dst_y1 = max(0, dy)
        dst_y2 = min(h, h + dy)
        src_x1 = max(0, -dx)
        src_x2 = min(w, w - dx)
        dst_x1 = max(0, dx)
        dst_x2 = min(w, w + dx)

        if src_y2 <= src_y1 or src_x2 <= src_x1:
            continue

        shifted[dst_y1:dst_y2, dst_x1:dst_x2] = core_uint8[src_y1:src_y2, src_x1:src_x2]

        shifted_shadow = (shifted > 0) & shadow_mask

        if not np.any(shifted_shadow):
            continue

        overlap_with_shadow = np.sum(shifted_shadow)
        total_shadow = np.sum(shadow_mask)
        shadow_coverage = overlap_with_shadow / total_shadow if total_shadow > 0 else 0

        if shadow_coverage > 0.25:
            shadow_pixels = gray[shifted_shadow]
            core_pixels = gray[core_text_mask]
            mean_shadow = np.mean(shadow_pixels)
            mean_core = np.mean(core_pixels)

            # Shadow should be brighter than core text (closer to background)
            if mean_shadow <= mean_core:
                continue

            bg_border_mask = np.zeros_like(gray, dtype=bool)
            border_size = 2
            bg_border_mask[:border_size, :] = True
            bg_border_mask[-border_size:, :] = True
            bg_border_mask[:, :border_size] = True
            bg_border_mask[:, -border_size:] = True
            bg_border_mask = bg_border_mask & ~all_dark_mask

            if np.any(bg_border_mask):
                mean_bg = np.mean(gray[bg_border_mask])
            else:
                mean_bg = 240.0

            shadow_bg_diff = mean_bg - mean_shadow

            shadow_candidates.append({
                "dx": float(dx),
                "dy": float(dy),
                "blur": float(max(2, dx)),
                "darkness": shadow_bg_diff,
                "ratio": shadow_coverage,
            })

    if not shadow_candidates:
        return None

    best = max(shadow_candidates, key=lambda s: s["darkness"] * s["ratio"])

    # Require strong evidence for shadow
    if best["darkness"] < 30:
        return None
    if best["ratio"] < 0.30:
        return None
    if best["darkness"] * best["ratio"] < 15:
        return None

    shadow_color = (0, 0, 0, 100)

    return {
        "dx": best["dx"],
        "dy": best["dy"],
        "blur": best["blur"],
        "color": shadow_color,
    }


def _detect_alignment(region_w: int, region_h: int, crop: np.ndarray) -> Optional[str]:
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape
    if h < 10 or w < 10:
        return None

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_count = np.sum(binary == 0)
    bright_count = np.sum(binary == 255)
    text_mask = binary == 0 if dark_count <= bright_count else binary == 255

    if not np.any(text_mask):
        return None

    col_sums = np.sum(text_mask, axis=0)
    total_fg = np.sum(col_sums)
    if total_fg == 0:
        return None

    col_weights = col_sums / total_fg
    col_indices = np.arange(w)
    center_of_mass = np.sum(col_weights * col_indices)

    rel_center = center_of_mass / w

    if rel_center > 0.55:
        return "right"
    elif rel_center < 0.45:
        return "left"
    else:
        return "center"


def _detect_v_align(crop: np.ndarray) -> Optional[dict]:
    """Detect vertical text alignment and precise offset within the crop region.

    Returns dict with: align ("top"/"center"/"bottom"), offset_ratio (0=top, 1=bottom)
    """
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape
    if h < 10 or w < 10:
        return None

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_count = np.sum(binary == 0)
    bright_count = np.sum(binary == 255)
    text_mask = binary == 0 if dark_count <= bright_count else binary == 255

    if not np.any(text_mask):
        return None

    row_sums = np.sum(text_mask, axis=1)
    total_fg = np.sum(row_sums)
    if total_fg == 0:
        return None

    row_weights = row_sums / total_fg
    row_indices = np.arange(h)
    center_of_mass = np.sum(row_weights * row_indices)
    rel_center = center_of_mass / h

    ys, _ = np.where(text_mask)
    text_top = ys.min()
    text_bottom = ys.max()
    text_height = text_bottom - text_top + 1

    # Offset ratio: where should the new text's top be placed?
    # 0 = top of crop, 1 = bottom of crop minus text height
    available_space = max(h - text_height, 1)
    offset_ratio = text_top / available_space if available_space > 0 else 0.0
    offset_ratio = max(0.0, min(1.0, offset_ratio))

    if rel_center > 0.55:
        align = "bottom"
    elif rel_center < 0.45:
        align = "top"
    else:
        align = "center"

    return {"align": align, "offset_ratio": offset_ratio}
