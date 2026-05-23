from __future__ import annotations
import logging

import cv2
import numpy as np

from backend.models.region import TextRegion

logger = logging.getLogger(__name__)


def detect_mser_regions(
    image: np.ndarray,
    existing_regions: list[TextRegion],
    min_area: int = 100,
    max_area_ratio: float = 0.05,
) -> list[TextRegion]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(int(h * w * max_area_ratio))

    try:
        regions, _ = mser.detectRegions(gray)
    except Exception:
        regions = []

    candidates = []

    for region in regions:
        if len(region) < 5:
            continue

        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        x, y, bw, bh = cv2.boundingRect(hull)

        if bw < 10 or bh < 10:
            continue

        if bw > w * 0.7 or bh > h * 0.5:
            continue

        area = cv2.contourArea(hull)
        rect_area = bw * bh
        if rect_area == 0:
            continue
        solidity = area / rect_area
        if solidity < 0.4:
            continue

        aspect_ratio = bw / bh
        if aspect_ratio > 10 or aspect_ratio < 0.2:
            continue

        sw = _compute_stroke_width(gray, region)
        if sw > 0:
            sw_var = _stroke_width_variance(region, sw)
            if sw_var > 0.8:
                continue

        if _overlaps_existing(x, y, bw, bh, existing_regions, threshold=0.4):
            continue

        is_tiny = bh < h * 0.035 or bw < w * 0.08 or bh < 24

        candidate = TextRegion.create(
            polygon=[[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
            bbox=[x, y, x + bw, y + bh],
            angle=0.0,
            source="mser_detection",
            confidence=0.4,
            is_tiny=is_tiny,
        )
        candidate.status = "suspected_text"
        candidates.append(candidate)

    merged = _merge_nearby_candidates(candidates, h, w)

    filtered = []
    for r in merged:
        rx1, ry1, rx2, ry2 = r.bbox
        rw = rx2 - rx1
        rh = ry2 - ry1

        if rw > w * 0.5 and rh > h * 0.3:
            continue

        if rw * rh > w * h * 0.05 and rh > h * 0.3:
            continue

        aspect = rw / rh if rh > 0 else 0
        if rw > 200 and rh > 200 and 0.5 < aspect < 2.0:
            continue

        filtered.append(r)

    return filtered


def detect_swt_regions(
    image: np.ndarray,
    existing_regions: list[TextRegion],
) -> list[TextRegion]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    swt_map = _compute_swt(gray)
    if swt_map is None:
        return []

    binary = (swt_map > 0).astype(np.uint8) * 255

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_v)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 40:
            continue
        if bw < 8 or bh < 8:
            continue

        if bw > w * 0.7 or bh > h * 0.5:
            continue

        aspect_ratio = bw / bh if bh > 0 else 0
        if aspect_ratio > 15 or aspect_ratio < 0.15:
            continue

        rect_area = bw * bh
        if rect_area == 0:
            continue
        fill_ratio = area / rect_area
        if fill_ratio < 0.15:
            continue

        if _overlaps_existing(x, y, bw, bh, existing_regions, threshold=0.4):
            continue

        is_tiny = bh < h * 0.035 or bw < w * 0.08 or bh < 24

        candidate = TextRegion.create(
            polygon=[[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
            bbox=[x, y, x + bw, y + bh],
            angle=0.0,
            source="swt_detection",
            confidence=0.35,
            is_tiny=is_tiny,
        )
        candidate.status = "suspected_text"
        candidates.append(candidate)

    merged = _merge_nearby_candidates(candidates, h, w)
    return merged


def _compute_swt(gray: np.ndarray) -> np.ndarray | None:
    edges = cv2.Canny(gray, 50, 150)
    if edges is None:
        return None

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude[magnitude == 0] = 1.0
    grad_x = sobel_x / magnitude
    grad_y = sobel_y / magnitude

    h, w = gray.shape
    swt = np.zeros((h, w), dtype=np.float32)

    edge_points = np.where(edges > 0)
    if len(edge_points[0]) == 0:
        return None

    for idx in range(len(edge_points[0])):
        y, x = edge_points[0][idx], edge_points[1][idx]
        dx, dy = grad_x[y, x], grad_y[y, x]

        for direction in [1, -1]:
            step = 0
            max_steps = max(h, w)
            while step < max_steps:
                step += 1
                px = int(round(x + direction * dx * step))
                py = int(round(y + direction * dy * step))

                if px < 0 or px >= w or py < 0 or py >= h:
                    break

                dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5

                if edges[py, px] > 0:
                    opp_dx, opp_dy = grad_x[py, px], grad_y[py, px]
                    cos_angle = dx * opp_dx + dy * opp_dy
                    if cos_angle < -0.1:
                        stroke_w = dist
                        for s in range(1, int(dist) + 1):
                            ix = int(round(x + direction * dx * s))
                            iy = int(round(y + direction * dy * s))
                            if 0 <= ix < w and 0 <= iy < h:
                                if swt[iy, ix] == 0:
                                    swt[iy, ix] = stroke_w
                                else:
                                    swt[iy, ix] = min(swt[iy, ix], stroke_w)
                    break

    return swt


def _compute_stroke_width(gray: np.ndarray, region: np.ndarray) -> float:
    if len(region) < 2:
        return 0.0

    x_coords = region[:, 0]
    y_coords = region[:, 1]
    x1, y1 = int(x_coords.min()), int(y_coords.min())
    x2, y2 = int(x_coords.max()), int(y_coords.max())

    h, w = gray.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = gray[y1:y2, x1:x2]
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    fg_pixels = dist[dist > 0]
    if len(fg_pixels) == 0:
        return 0.0

    return float(np.median(fg_pixels)) * 2


def _stroke_width_variance(region: np.ndarray, median_sw: float) -> float:
    if len(region) < 2 or median_sw <= 0:
        return 0.0

    x_coords = region[:, 0]
    y_coords = region[:, 1]

    distances = np.sqrt(
        (x_coords - x_coords.mean()) ** 2 + (y_coords - y_coords.mean()) ** 2
    )

    if len(distances) == 0:
        return 0.0

    return float(np.std(distances) / (np.mean(distances) + 1e-6))


def _overlaps_existing(
    x: int, y: int, bw: int, bh: int,
    existing_regions: list[TextRegion],
    threshold: float = 0.4,
) -> bool:
    for region in existing_regions:
        if not region.bbox or len(region.bbox) < 4:
            continue

        rx1, ry1, rx2, ry2 = region.bbox
        ix1 = max(x, rx1)
        iy1 = max(y, ry1)
        ix2 = min(x + bw, rx2)
        iy2 = min(y + bh, ry2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = bw * bh

        if area > 0 and inter / area > threshold:
            return True

    return False


def _merge_nearby_candidates(
    candidates: list[TextRegion], img_h: int, img_w: int,
) -> list[TextRegion]:
    if not candidates:
        return []

    h, w = img_h, img_w

    lines = _group_into_lines(candidates, h)

    merged = []
    for line in lines:
        if len(line) == 1:
            merged.append(line[0])
            continue

        line.sort(key=lambda r: r.bbox[0])

        groups = _split_line_into_words(line, h)

        for group in groups:
            all_x1 = [r.bbox[0] for r in group]
            all_y1 = [r.bbox[1] for r in group]
            all_x2 = [r.bbox[2] for r in group]
            all_y2 = [r.bbox[3] for r in group]

            combined = TextRegion.create(
                polygon=[
                    [min(all_x1), min(all_y1)],
                    [max(all_x2), min(all_y1)],
                    [max(all_x2), max(all_y2)],
                    [min(all_x1), max(all_y2)],
                ],
                bbox=[min(all_x1), min(all_y1), max(all_x2), max(all_y2)],
                angle=0.0,
                source="mser_merged" if any("mser" in r.source for r in group) else "swt_merged",
                confidence=max(r.confidence for r in group),
                is_tiny=any(r.is_tiny for r in group),
            )
            combined.status = "suspected_text"
            merged.append(combined)

    return merged


def _group_into_lines(candidates: list[TextRegion], img_h: int) -> list[list[TextRegion]]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda r: (r.bbox[1], r.bbox[0]))

    lines = []
    used = [False] * len(sorted_candidates)

    for i, c1 in enumerate(sorted_candidates):
        if used[i]:
            continue

        line = [c1]
        used[i] = True

        cy1 = c1.bbox[1]
        cy2 = c1.bbox[3]
        ch = cy2 - cy1

        for j, c2 in enumerate(sorted_candidates):
            if used[j]:
                continue

            ay1 = c2.bbox[1]
            ay2 = c2.bbox[3]
            ah = ay2 - ay1

            center_i = (cy1 + cy2) / 2
            center_j = (ay1 + ay2) / 2

            max_h = max(ch, ah)
            v_dist = abs(center_i - center_j)

            if v_dist < max_h * 0.5:
                line.append(c2)
                used[j] = True

        lines.append(line)

    return lines


def _split_line_into_words(line: list[TextRegion], img_h: int) -> list[list[TextRegion]]:
    if len(line) <= 1:
        return [line]

    avg_height = np.mean([r.bbox[3] - r.bbox[1] for r in line])

    gaps = []
    for i in range(len(line) - 1):
        gap = line[i + 1].bbox[0] - line[i].bbox[2]
        gaps.append(gap)

    if not gaps:
        return [line]

    gap_threshold = max(avg_height * 0.8, np.median(gaps) * 2.0) if gaps else avg_height

    groups = []
    current_group = [line[0]]

    for i in range(1, len(line)):
        gap = line[i].bbox[0] - line[i - 1].bbox[2]
        if gap > gap_threshold:
            groups.append(current_group)
            current_group = [line[i]]
        else:
            current_group.append(line[i])

    groups.append(current_group)
    return groups
