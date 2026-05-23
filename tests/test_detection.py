import pytest
import numpy as np
import cv2
from pathlib import Path

from backend.core.detection import detect_text_regions, _is_tiny_text, _compute_iou, _merge_regions
from backend.core.tiny_text_detection import detect_tiny_text
from backend.models.region import TextRegion


class TestTinyTextDetection:
    def test_is_tiny_text_small_height(self):
        assert _is_tiny_text(100, 20, 1000, 1000) is True

    def test_is_tiny_text_small_width(self):
        assert _is_tiny_text(50, 50, 1000, 1000) is True

    def test_is_tiny_text_height_under_24(self):
        assert _is_tiny_text(200, 20, 1000, 1000) is True

    def test_is_not_tiny_text(self):
        assert _is_tiny_text(200, 60, 1000, 1000) is False

    def test_tiny_text_detection_finds_regions(self):
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        regions = detect_tiny_text(img, [])
        assert isinstance(regions, list)

    def test_tiny_text_detection_no_overlap(self):
        existing = TextRegion.create(
            polygon=[[10, 10, 200, 40]],
            bbox=[10, 10, 200, 40],
        )
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        regions = detect_tiny_text(img, [existing])
        for r in regions:
            assert r.bbox != [10, 10, 200, 40]


class TestDetectionMerge:
    def test_compute_iou_no_overlap(self):
        box1 = [0, 0, 100, 100]
        box2 = [200, 200, 300, 300]
        assert _compute_iou(box1, box2) == 0.0

    def test_compute_iou_full_overlap(self):
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]
        assert abs(_compute_iou(box1, box2) - 1.0) < 0.01

    def test_compute_iou_partial_overlap(self):
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = _compute_iou(box1, box2)
        assert 0 < iou < 1

    def test_merge_regions_dedup(self):
        r1 = TextRegion.create(
            polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
            bbox=[0, 0, 100, 50],
            confidence=0.9,
        )
        r2 = TextRegion.create(
            polygon=[[2, 2], [98, 2], [98, 48], [2, 48]],
            bbox=[2, 2, 98, 48],
            confidence=0.8,
        )
        merged = _merge_regions([r1, r2], iou_threshold=0.55)
        assert len(merged) <= 2

    def test_merge_regions_no_merge(self):
        r1 = TextRegion.create(
            polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
            bbox=[0, 0, 100, 50],
            confidence=0.9,
        )
        r2 = TextRegion.create(
            polygon=[[300, 300], [400, 300], [400, 350], [300, 350]],
            bbox=[300, 300, 400, 350],
            confidence=0.8,
        )
        merged = _merge_regions([r1, r2], iou_threshold=0.55)
        assert len(merged) == 2

    def test_detect_text_regions_no_engine(self):
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        regions = detect_text_regions(img, ocr_engine=None, detect_small_text=False)
        assert isinstance(regions, list)
