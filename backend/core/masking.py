from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend.models.region import TextRegion

logger = logging.getLogger(__name__)


def generate_masks(
    image: np.ndarray, region: TextRegion,
    use_grabcut: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not region.bbox or len(region.bbox) < 4:
        h, w = image.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty, empty

    x1, y1, x2, y2 = [int(v) for v in region.bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty, empty

    crop = image[y1:y2, x1:x2]
    glyph_mask = _generate_glyph_mask(crop)

    glyph_full = np.zeros((h, w), dtype=np.uint8)
    glyph_full[y1:y2, x1:x2] = glyph_mask

    if use_grabcut:
        refined_mask = _refine_with_grabcut(image, glyph_full, x1, y1, x2, y2)
    else:
        refined_mask = glyph_full

    erase_mask = _generate_erase_mask(refined_mask, x1, y1, x2, y2, region)

    safe_mask = _generate_safe_mask(x1, y1, x2, y2, w, h)

    return refined_mask, erase_mask, safe_mask


def _generate_glyph_mask(crop: np.ndarray) -> np.ndarray:
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros((h, w), dtype=np.uint8)

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu_mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    block_size = max(3, min(w, h) // 4 * 2 + 1)
    adaptive_mask = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 3
    )

    binary = cv2.bitwise_or(otsu_mask, adaptive_mask)

    mean_val = np.mean(gray)
    if mean_val < 128:
        binary = cv2.bitwise_not(binary)

    total_pixels = h * w
    fg_ratio = np.sum(binary > 0) / total_pixels

    if fg_ratio > 0.6:
        binary = cv2.bitwise_not(binary)
        fg_ratio = 1.0 - fg_ratio

    if fg_ratio < 0.01:
        return np.zeros_like(binary)

    k_close = min(5, max(1, min(w, h) // 20))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, max(1, k_close // 2)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    k_open = min(3, k_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, max(1, k_open // 2)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    return binary


def _refine_with_grabcut(
    image: np.ndarray,
    glyph_mask: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    region_h = y2 - y1
    region_w = x2 - x1

    if region_w < 20 or region_h < 20:
        return glyph_mask

    if len(image.shape) == 2:
        gc_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        gc_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        gc_image = image.copy()

    pad = 5
    gx1 = max(0, x1 - pad)
    gy1 = max(0, y1 - pad)
    gx2 = min(w, x2 + pad)
    gy2 = min(h, y2 + pad)

    crop_img = gc_image[gy1:gy2, gx1:gx2]
    crop_mask = glyph_mask[gy1:gy2, gx1:gx2]

    gc_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
    gc_mask[crop_mask > 0] = cv2.GC_FGD
    gc_mask[crop_mask == 0] = cv2.GC_PR_BGD

    inner_x1 = min(pad, gx2 - gx1)
    inner_y1 = min(pad, gy2 - gy1)
    inner_x2 = max(0, crop_img.shape[1] - pad)
    inner_y2 = max(0, crop_img.shape[0] - pad)

    border_mask = np.zeros_like(gc_mask)
    border_mask[:inner_y1, :] = cv2.GC_BGD
    border_mask[inner_y2:, :] = cv2.GC_BGD
    border_mask[:, :inner_x1] = cv2.GC_BGD
    border_mask[:, inner_x2:] = cv2.GC_BGD

    gc_mask[border_mask == cv2.GC_BGD] = cv2.GC_BGD
    gc_mask[(gc_mask != cv2.GC_FGD) & (gc_mask != cv2.GC_BGD) & (border_mask != cv2.GC_BGD)] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(crop_img, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        refined_crop = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        original_fg_count = np.sum(crop_mask > 0)
        refined_fg_count = np.sum(refined_crop > 0)

        if refined_fg_count > original_fg_count * 3 or refined_fg_count < original_fg_count * 0.2:
            refined_crop = crop_mask
    except Exception:
        refined_crop = crop_mask

    refined_full = np.zeros((h, w), dtype=np.uint8)
    refined_full[gy1:gy2, gx1:gx2] = refined_crop

    refined_full[:y1, :] = 0
    refined_full[y2:, :] = 0
    refined_full[:, :x1] = 0
    refined_full[:, x2:] = 0

    return refined_full


def _generate_erase_mask(
    glyph_mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, region: TextRegion
) -> np.ndarray:
    """Generate precise erase mask covering only text pixels + minimal AA fringe.

    The glyph mask captures solid text; 1px dilation covers the nearest AA fringe.
    Additional expansion only for explicitly detected stroke/shadow.
    """
    erase = glyph_mask.copy()

    has_shadow = bool(region.style and region.style.shadow)
    has_stroke = bool(
        region.style and region.style.stroke_width > 0 and region.style.stroke_color
    )

    # 1px dilation to cover the nearest anti-aliasing fringe
    kernel_aa = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erase = cv2.dilate(erase, kernel_aa, iterations=1)

    # Stroke: targeted expansion by detected stroke width
    if has_stroke and region.style:
        stroke_w = max(1, int(region.style.stroke_width))
        if stroke_w > 1:
            stroke_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (stroke_w * 2 + 1, stroke_w * 2 + 1)
            )
            erase = cv2.dilate(erase, stroke_kernel, iterations=1)

    # Shadow: targeted expansion in shadow direction only
    if has_shadow and region.style and region.style.shadow:
        sd = region.style.shadow
        shadow_dx = int(abs(sd.dx)) + 1
        shadow_dy = int(abs(sd.dy)) + 1
        if shadow_dx > 1 or shadow_dy > 1:
            shadow_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (shadow_dx * 2 + 1, shadow_dy * 2 + 1),
            )
            shadow_mask = cv2.dilate(glyph_mask, shadow_kernel, iterations=1)
            erase = cv2.bitwise_or(erase, shadow_mask)

    # Clip to bbox
    erase[:y1, :] = 0
    erase[y2:, :] = 0
    erase[:, :x1] = 0
    erase[:, x2:] = 0

    return erase


def _generate_safe_mask(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> np.ndarray:
    margin = 5
    sx1 = max(0, x1 - margin)
    sy1 = max(0, y1 - margin)
    sx2 = min(img_w, x2 + margin)
    sy2 = min(img_h, y2 + margin)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[sy1:sy2, sx1:sx2] = 255

    return mask


def classify_background(image: np.ndarray, region: TextRegion) -> str:
    if not region.bbox or len(region.bbox) < 4:
        return "solid"

    x1, y1, x2, y2 = [int(v) for v in region.bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return "solid"

    region_h = y2 - y1
    region_w = x2 - x1

    # Small regions: border-sampling fill is more reliable than OpenCV inpainting
    if region_h < 20 or region_w < 30:
        return "solid"

    crop = image[y1:y2, x1:x2]

    margin_x = max(2, int(region_w * 0.05))
    margin_y = max(2, int(region_h * 0.1))
    inner = crop[margin_y:crop.shape[0]-margin_y, margin_x:crop.shape[1]-margin_x]
    if inner.size == 0:
        inner = crop

    variance = _color_variance(inner)
    edge_density_value = _edge_density(inner)
    gradient_score = _gradient_smoothness(inner)

    if variance < 8 and edge_density_value < 0.03:
        return "solid"

    if gradient_score > 0.75 and edge_density_value < 0.08:
        return "gradient"

    if edge_density_value > 0.15:
        return "complex"

    return "texture"


def _color_variance(crop: np.ndarray) -> float:
    if len(crop.shape) == 2:
        return float(np.var(crop))
    return float(np.mean([np.var(crop[:, :, c]) for c in range(crop.shape[2])]))


def _edge_density(crop: np.ndarray) -> float:
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0)) / (edges.shape[0] * edges.shape[1])


def _gradient_smoothness(crop: np.ndarray) -> float:
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if magnitude.max() == 0:
        return 1.0

    direction = np.arctan2(grad_y, grad_x)
    direction_var = np.var(direction[magnitude > magnitude.mean()])

    return float(max(0, 1.0 - direction_var / (np.pi**2)))
