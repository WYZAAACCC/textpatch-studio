from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np

from backend.models.region import TextRegion
from backend.core.preprocessing import preprocess_image

logger = logging.getLogger(__name__)


def detect_text_regions(
    image: np.ndarray,
    ocr_engine=None,
    scales: list = None,
    detect_small_text: bool = True,
    language: str = "zh-CN",
) -> list[TextRegion]:
    if scales is None:
        scales = [1, 2, 3]

    h, w = image.shape[:2]
    all_regions = []

    if ocr_engine is not None:
        _, img_encoded = cv2.imencode(".png", image)
        image_bytes = img_encoded.tobytes()

        try:
            ocr_results = ocr_engine.detect_and_recognize(image_bytes, file_type=1)
            for ocr_res in ocr_results:
                bbox = ocr_res.bbox
                if not bbox or len(bbox) < 4:
                    continue

                if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                    x1, y1, x2, y2 = bbox
                    polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                elif len(bbox) >= 4 and isinstance(bbox[0], list):
                    polygon = bbox[:4]
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                else:
                    continue

                region_h = y2 - y1
                region_w = x2 - x1

                # Filter noise: very small or low-confidence OCR detections
                if region_w < 12 and region_h < 12:
                    continue
                if ocr_res.confidence < 0.4:
                    continue
                area = region_w * region_h
                if area < 200 and ocr_res.confidence < 0.6:
                    continue

                is_tiny = _is_tiny_text(region_w, region_h, w, h)

                region = TextRegion.create(
                    polygon=polygon,
                    bbox=[x1, y1, x2, y2],
                    angle=ocr_res.angle if hasattr(ocr_res, "angle") else 0.0,
                    source="paddle_ocr",
                    confidence=ocr_res.confidence,
                    is_tiny=is_tiny,
                )
                region.final_text = ocr_res.text
                all_regions.append(region)

        except Exception as e:
            logger.error(f"OCR detection failed: {e}")

    if detect_small_text:
        from backend.core.tiny_text_detection import detect_tiny_text
        tiny_regions = detect_tiny_text(image, all_regions)
        all_regions.extend(tiny_regions)

        from backend.core.mser_swt_detection import detect_mser_regions, detect_swt_regions
        mser_regions = detect_mser_regions(image, all_regions)
        all_regions.extend(mser_regions)

        swt_regions = detect_swt_regions(image, all_regions)
        all_regions.extend(swt_regions)

    merged = _merge_regions(all_regions, iou_threshold=0.55, angle_threshold=8.0)

    for region in merged:
        if region.is_tiny and not region.final_text:
            region.status = "suspected_text"

    return merged


def _is_tiny_text(box_w: float, box_h: float, img_w: int, img_h: int) -> bool:
    return box_h < img_h * 0.035 or box_w < img_w * 0.08 or box_h < 24


def _compute_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _merge_regions(
    regions: list[TextRegion], iou_threshold: float = 0.55, angle_threshold: float = 8.0
) -> list[TextRegion]:
    if not regions:
        return []

    merged = []
    used = set()

    for i, r1 in enumerate(regions):
        if i in used:
            continue
        group = [r1]
        used.add(i)

        for j, r2 in enumerate(regions):
            if j in used:
                continue

            iou = _compute_iou(r1.bbox, r2.bbox)
            if iou > iou_threshold:
                c1 = _bbox_center(r1.bbox)
                c2 = _bbox_center(r2.bbox)
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                angle_diff = abs(r1.angle - r2.angle)

                if dist < max(_bbox_size(r1.bbox)) and angle_diff < angle_threshold:
                    group.append(r2)
                    used.add(j)

        best = max(group, key=lambda r: (r.confidence, _bbox_area(r.bbox)))
        if best.is_tiny:
            high_scale = [r for r in group if r.source and "2x" in r.source or "3x" in r.source]
            if high_scale:
                best = max(high_scale, key=lambda r: r.confidence)

        if best.final_text:
            for r in group:
                if r.final_text and r is not best:
                    if r.confidence > best.confidence * 0.9:
                        if len(r.final_text) > len(best.final_text):
                            best = r

        merged.append(best)

    return merged


def _bbox_center(bbox: list) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_size(bbox: list) -> tuple:
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _bbox_area(bbox: list) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
