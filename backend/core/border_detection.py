"""Classify whether a detected region is a structural element (border,
line, box frame) rather than actual text.

Scientific flowcharts have dark border lines and box frames around
coloured blocks. Detectors (especially MSER/SWT) pick these up as
"text" regions. Erasing them cuts holes in the image structure.
"""

from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np

from backend.models.region import TextRegion

logger = logging.getLogger(__name__)


def is_border_region(region: TextRegion, image: np.ndarray) -> bool:
    """Classify whether a detected region is a structural element.

    Uses three levels of checks:
    1. OCR text content — regions with substantial text are NOT borders.
    2. Geometric features — elongated strips with line-like properties.
    3. Structural continuation — lines that extend beyond the bbox.
    """
    if not region.bbox or len(region.bbox) < 4:
        return False

    # ---- Level 1: OCR text content ----
    # Regions with substantial OCR text are NOT borders, period.
    text = (region.final_text or "").strip()
    if len(text) >= 10:
        return False

    # Grouped regions are merged OCR results — always real text.
    if region.source and "grouped" in region.source:
        return False

    x1, y1, x2, y2 = [int(v) for v in region.bbox]
    h_img, w_img = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    region_w = x2 - x1
    region_h = y2 - y1

    if region_w <= 0 or region_h <= 0:
        return False

    area = region_w * region_h
    if area > 30000:
        return False
    if region_w < 8 or region_h < 8:
        return False

    # Crop the region
    crop = image[y1:y2, x1:x2]
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    dark_pixel_count = np.sum(binary > 0)
    total_pixels = region_w * region_h
    fill_ratio = dark_pixel_count / total_pixels if total_pixels > 0 else 0

    if fill_ratio > 0.70 or fill_ratio < 0.02:
        return False

    aspect_ratio = max(region_w, region_h) / (min(region_w, region_h) + 1)

    # ---- Level 2: Geometric line detection ----
    # Compute straightness once — used by multiple checks
    straightness = _compute_straightness(binary, region_w, region_h)

    # Check 2a: MSER/SWT/tiny regions — these detectors produce many
    # false positives on structural lines and box frames.
    if region.source and "paddle" not in region.source and "grouped" not in (region.source or ""):
        # Elongated strips: thin line-like regions
        if aspect_ratio >= 2.0 and fill_ratio < 0.50:
            if straightness > 0.78:
                return True
        # Very thin regions with any straightness
        min_dim = min(region_w, region_h)
        if min_dim <= 18 and fill_ratio < 0.50 and straightness > 0.75:
            return True
        # Very short text on thin regions — likely structural fragment.
        # Require elongation or extreme thinness to avoid catching small
        # square-ish text regions (e.g. 1-2 char Chinese text).
        if len(text) <= 2 and straightness > 0.80:
            if aspect_ratio >= 2.0 and min_dim <= 40:
                return True
            if min_dim <= 15:
                return True
        if _is_box_frame(gray, binary, region_w, region_h, fill_ratio):
            return True

    # Check 2b: PaddleOCR regions with extreme line geometry
    # PaddleOCR can detect structural lines and assign short/garbled
    # OCR text to them. These have very specific geometric signatures.
    if region.source and "paddle" in region.source:
        if _is_structural_line(
            gray, binary, region_w, region_h,
            aspect_ratio, fill_ratio, straightness,
            text, image, x1, y1,
        ):
            return True

    # Check 2c: Thin edge-aligned region (all sources)
    min_dim = min(region_w, region_h)
    if min_dim <= 8 and fill_ratio < 0.50 and len(text) < 2:
        if _aligns_with_edge(gray, region_w, region_h):
            return True

    return False


def _is_structural_line(
    gray: np.ndarray,
    binary: np.ndarray,
    w: int,
    h: int,
    aspect_ratio: float,
    fill_ratio: float,
    straightness: float,
    text: str,
    image: np.ndarray,
    x1: int,
    y1: int,
) -> bool:
    """Check if a PaddleOCR region is actually a structural line.

    PaddleOCR sometimes detects dark border lines, box edges, and
    arrows as text regions, assigning 1-9 garbled characters to them.

    Key discriminators vs real short text:
    - Structural lines have foregound pixels that are uniformly dark
    - Text characters have varied foreground (anti-aliasing, different strokes)
    - Structural lines continue beyond the detected bbox
    - Text ends at the bbox boundary
    """
    # Must be elongated
    if aspect_ratio < 4.0:
        return False

    # Must have low fill (hollow structure)
    if fill_ratio > 0.45:
        return False

    # Must be very straight
    if straightness < 0.90:
        return False

    # ---- Discriminator 1: Foreground color uniformity ----
    # Structural lines: all dark pixels have very similar intensity
    # Text: dark pixels vary (character edges, anti-aliasing)
    fg_mask = binary > 0
    fg_pixels = gray[fg_mask]
    if len(fg_pixels) < 10:
        return False

    fg_std = float(np.std(fg_pixels))
    fg_mean = float(np.mean(fg_pixels))

    # Structural lines have uniform dark foreground (low std)
    # Text has varied foreground (higher std from character shapes)
    if fg_std > 25:
        return False

    # Structural lines are dark
    if fg_mean > 100:
        return False

    # ---- Discriminator 2: Structural continuation ----
    # Structural lines extend beyond the bbox along their primary axis.
    # Text ends where the characters end.
    if _has_structural_continuation(
        image, binary, w, h, x1, y1, aspect_ratio
    ):
        return True

    # ---- Discriminator 3: Very short text on extreme geometry ----
    # If the OCR text is very short (< 5 chars) AND the geometry is
    # extreme, it's very likely a structural element misdetected as text.
    if len(text) < 5 and aspect_ratio >= 6.0 and fill_ratio < 0.30:
        return True

    return False


def _has_structural_continuation(
    image: np.ndarray,
    binary: np.ndarray,
    w: int,
    h: int,
    x1: int,
    y1: int,
    aspect_ratio: float,
) -> bool:
    """Check if dark pixels continue beyond the region bbox along the line.

    A structural line (border, arrow, separator) continues past the
    detected region. Text does not — it ends where the characters end.
    """
    h_img, w_img = image.shape[:2]
    gray_full = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image
    )

    # Determine primary axis direction from aspect ratio
    is_horizontal = w >= h

    # Check both ends of the line
    continuation_score = 0

    if is_horizontal:
        # Check left and right extensions
        ext_len = min(w // 3, 30)
        if ext_len < 5:
            return False

        # Left extension
        left_x1 = max(0, x1 - ext_len)
        left_x2 = x1
        if left_x2 > left_x1:
            left_strip = gray_full[y1:y1+h, left_x1:left_x2]
            # Check if there are dark pixels continuing
            dark_count = np.sum(left_strip < 80)
            total = left_strip.size
            if total > 10:
                continuation_score += dark_count / total

        # Right extension
        right_x1 = x1 + w
        right_x2 = min(w_img, x1 + w + ext_len)
        if right_x2 > right_x1:
            right_strip = gray_full[y1:y1+h, right_x1:right_x2]
            dark_count = np.sum(right_strip < 80)
            total = right_strip.size
            if total > 10:
                continuation_score += dark_count / total
    else:
        # Check top and bottom extensions
        ext_len = min(h // 3, 30)
        if ext_len < 5:
            return False

        # Top extension
        top_y1 = max(0, y1 - ext_len)
        top_y2 = y1
        if top_y2 > top_y1:
            top_strip = gray_full[top_y1:top_y2, x1:x1+w]
            dark_count = np.sum(top_strip < 80)
            total = top_strip.size
            if total > 10:
                continuation_score += dark_count / total

        # Bottom extension
        bottom_y1 = y1 + h
        bottom_y2 = min(h_img, y1 + h + ext_len)
        if bottom_y2 > bottom_y1:
            bottom_strip = gray_full[bottom_y1:bottom_y2, x1:x1+w]
            dark_count = np.sum(bottom_strip < 80)
            total = bottom_strip.size
            if total > 10:
                continuation_score += dark_count / total

    # If > 15% of extension pixels are dark, the line continues
    return continuation_score > 0.30


def _is_box_frame(
    gray: np.ndarray,
    binary: np.ndarray,
    w: int,
    h: int,
    fill_ratio: float,
) -> bool:
    """Check if the region is a box/frame border segment.

    Box frames have strong edges along the perimeter but low edge
    density inside (hollow interior).
    """
    if w < 12 or h < 12:
        return False

    edges = cv2.Canny(gray, 30, 100)

    # Perimeter edge density
    perimeter_mask = np.zeros((h, w), dtype=np.uint8)
    perimeter_mask[0, :] = 1
    perimeter_mask[-1, :] = 1
    perimeter_mask[:, 0] = 1
    perimeter_mask[:, -1] = 1
    perimeter_edges = np.sum(edges[perimeter_mask > 0])
    perimeter_total = np.sum(perimeter_mask)
    perimeter_density = perimeter_edges / perimeter_total if perimeter_total > 0 else 0

    # Interior edge density
    margin = 3
    if w <= 2 * margin or h <= 2 * margin:
        return False
    interior = edges[margin:-margin, margin:-margin]
    interior_density = np.sum(interior > 0) / interior.size if interior.size > 0 else 0

    if perimeter_density < 0.15 or interior_density > 0.25:
        return False

    # Foreground concentration near edges
    edge_zone = np.zeros((h, w), dtype=bool)
    edge_zone[:margin, :] = True
    edge_zone[-margin:, :] = True
    edge_zone[:, :margin] = True
    edge_zone[:, -margin:] = True

    inner_zone = np.zeros((h, w), dtype=bool)
    if h > 2 * margin and w > 2 * margin:
        inner_zone[margin:-margin, margin:-margin] = True

    fg = binary > 0
    fg_total = np.sum(fg)
    if fg_total < 10:
        return False

    edge_fg_ratio = np.sum(fg & edge_zone) / fg_total
    inner_fg_ratio = np.sum(fg & inner_zone) / fg_total

    return (
        edge_fg_ratio > 0.55
        and inner_fg_ratio < 0.35
        and fill_ratio < 0.50
    )


def _aligns_with_edge(gray: np.ndarray, w: int, h: int) -> bool:
    """Check if a thin region aligns with strong image edges."""
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    return edge_density > 0.30


def _compute_straightness(binary: np.ndarray, region_w: int, region_h: int) -> float:
    """Measure how straight the foreground points are using PCA."""
    ys, xs = np.where(binary > 0)
    if len(xs) < 10:
        return 0.0

    points = np.column_stack([xs, ys]).astype(np.float64)
    mean = points.mean(axis=0)
    centered = points - mean

    try:
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    if len(s) < 2 or s[0] < 1e-6:
        return 0.0

    primary_variance_ratio = s[0] ** 2 / (s[0] ** 2 + s[1] ** 2) if len(s) >= 2 else 1.0

    primary_dir = vt[0, :]
    projections = centered @ primary_dir
    proj_range = projections.max() - projections.min()
    if proj_range < 1:
        return 0.0

    num_bins = min(20, max(5, int(proj_range / 3)))
    bin_edges = np.linspace(projections.min(), projections.max(), num_bins + 1)
    bin_counts = []
    for i in range(num_bins):
        count = np.sum((projections >= bin_edges[i]) & (projections < bin_edges[i + 1]))
        bin_counts.append(count)

    if not bin_counts:
        return 0.0

    bin_counts = np.array(bin_counts)
    non_empty_bins = np.sum(bin_counts > 0) / len(bin_counts)

    straightness = primary_variance_ratio * 0.6 + non_empty_bins * 0.4
    return float(np.clip(straightness, 0.0, 1.0))
