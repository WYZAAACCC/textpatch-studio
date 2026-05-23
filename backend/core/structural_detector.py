"""Detect structural diagram elements (lines, boxes, borders) to protect
them from text erasure in scientific flowchart images.

Scientific flowcharts have dark border lines, box frames, arrows, and
other structural elements that text detectors (PaddleOCR, MSER) may
misclassify as text. Erasing them damages the diagram.

This module builds a "structural mask" — pixels that must never be
erased — before inpainting runs.

Design principle: be conservative. Only protect pixels that are clearly
structural. Text on coloured backgrounds must not be blocked.
"""

from __future__ import annotations
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def build_structural_mask(
    image: np.ndarray,
    *,
    line_min_length: int = 30,
    line_threshold: int = 50,
    protection_radius: int = 1,
) -> np.ndarray:
    """Build a binary mask of structural pixels to protect from erasure.

    Only protects pixels that are clearly structural:
    1. Long straight lines (Hough transform on Canny edges)
    2. Thin box/rectangle frames (contour analysis)

    Returns a uint8 binary mask (255 = protected).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    structural = np.zeros((h, w), dtype=np.uint8)

    # ---- 1. Long straight lines via Hough transform ----
    # Use tight Canny thresholds to catch dark structural lines while
    # ignoring text edges.
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=line_threshold,
        minLineLength=line_min_length,
        maxLineGap=3,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw thin (1px) — dilation will add the protection buffer
            cv2.line(structural, (x1, y1), (x2, y2), 255, thickness=1)

    # ---- 2. Box / rectangle frames ----
    # Find hollow rectangular contours that form block borders.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) < 4 or len(approx) > 6:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 20 or bh < 20:
                continue
            if bw > w * 0.9 and bh > h * 0.9:
                continue  # Skip full-image border

            # Check if hollow (box frame, not filled rectangle)
            cnt_mask = np.zeros((bh + 2, bw + 2), dtype=np.uint8)
            shifted = cnt - [x, y]
            cv2.drawContours(cnt_mask, [shifted], -1, 255, 1)
            contour_pixels = np.sum(cnt_mask > 0)
            bbox_area = bw * bh
            contour_fill = contour_pixels / bbox_area

            # Box frames: perimeter is < 25% of bbox area
            # Filled blocks: perimeter is > 25%
            if contour_fill > 0.25:
                continue

            # Draw thin frame
            cv2.drawContours(structural, [cnt], -1, 255, thickness=1)

    # ---- 3. Dilate minimally to create protective buffer ----
    if protection_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (protection_radius * 2 + 1, protection_radius * 2 + 1),
        )
        structural = cv2.dilate(structural, kernel)

    return structural


def build_color_exclusion_mask(
    image: np.ndarray,
    structural_mask: np.ndarray,
) -> np.ndarray:
    """Build a broader mask of pixels to exclude from colour sampling.

    This is separate from the structural protection mask. We only protect
    truly structural pixels from erasure (to avoid damaging the diagram),
    but we can be more aggressive about excluding pixels from the colour
    ring — any dark, edge-heavy, or line-like pixel that is unlikely to
    be the true background should be excluded.

    Returns a uint8 binary mask (255 = exclude from colour sampling).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    exclusion = structural_mask.copy()

    # ---- Dark connected components ----
    # Dark pixels that form reasonably large connected components are
    # likely structural (lines, borders) or text, not background.
    _, dark_binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_binary)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_cc = stats[i, cv2.CC_STAT_WIDTH]
        h_cc = stats[i, cv2.CC_STAT_HEIGHT]
        # Exclude dark components that are large or elongated
        if area > 100 or max(w_cc, h_cc) > 30:
            exclusion[labels == i] = 255

    # ---- Edge pixels ----
    # Strong edge pixels are more likely to be structural than background.
    edges = cv2.Canny(gray, 50, 150)
    edge_dilated = cv2.dilate(edges, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    ))
    exclusion = cv2.bitwise_or(exclusion, edge_dilated)

    return exclusion
