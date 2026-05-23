from __future__ import annotations
import logging

import cv2
import numpy as np

from backend.models.region import TextRegion

logger = logging.getLogger(__name__)


def detect_tiny_text(
    image: np.ndarray, existing_regions: list[TextRegion]
) -> list[TextRegion]:
    h, w = image.shape[:2]
    tiny_regions = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Raised thresholds: fewer false positives, real text has more structure
        if area < 200:
            continue
        if bh < 14 or bw < 14:
            continue

        is_tiny = bh < h * 0.035 or bw < w * 0.08 or bh < 24
        if not is_tiny:
            continue

        if _overlaps_existing(x, y, bw, bh, existing_regions):
            continue

        aspect_ratio = bw / bh if bh > 0 else 0
        # Narrower aspect ratio: real text has constrained proportions
        if aspect_ratio < 0.5 or aspect_ratio > 12:
            continue

        region = TextRegion.create(
            polygon=[[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
            bbox=[x, y, x + bw, y + bh],
            angle=0.0,
            source="tiny_text_detection",
            confidence=0.3,
            is_tiny=True,
        )
        region.status = "suspected_text"
        tiny_regions.append(region)

    return tiny_regions


def _overlaps_existing(
    x: int, y: int, w: int, h: int, existing_regions: list[TextRegion], threshold: float = 0.5
) -> bool:
    for region in existing_regions:
        if not region.bbox or len(region.bbox) < 4:
            continue

        rx1, ry1, rx2, ry2 = region.bbox
        ix1 = max(x, rx1)
        iy1 = max(y, ry1)
        ix2 = min(x + w, rx2)
        iy2 = min(y + h, ry2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = w * h

        if area > 0 and inter / area > threshold:
            return True

    return False
