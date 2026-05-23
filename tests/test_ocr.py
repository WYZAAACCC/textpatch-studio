import pytest
import numpy as np
import cv2

from backend.models.region import TextRegion
from backend.models.ocr import OCRInfo, OCRCandidate
from backend.core.ocr_engine import run_ocr, _merge_candidates


class TestOCREngine:
    def test_run_ocr_no_engine(self):
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        regions = [
            TextRegion.create(
                polygon=[[10, 10], [200, 10], [200, 50], [10, 50]],
                bbox=[10, 10, 200, 50],
            )
        ]
        result = run_ocr(img, regions, ocr_engine=None)
        assert len(result) == 1
        assert result[0].ocr is None

    def test_run_ocr_empty_regions(self):
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        result = run_ocr(img, [], ocr_engine=None)
        assert result == []

    def test_merge_candidates_dedup(self):
        candidates = [
            OCRCandidate(text="hello", confidence=0.9, source="1x"),
            OCRCandidate(text="hello", confidence=0.8, source="2x"),
            OCRCandidate(text="hallo", confidence=0.7, source="3x"),
        ]
        merged = _merge_candidates(candidates)
        assert len(merged) == 2
        hello_candidates = [c for c in merged if c.text == "hello"]
        assert len(hello_candidates) == 1
        assert abs(hello_candidates[0].confidence - 0.85) < 0.01

    def test_ocr_info_to_dict(self):
        info = OCRInfo(
            best_text="test",
            confidence=0.9,
            candidates=[OCRCandidate(text="test", confidence=0.9, source="1x")],
        )
        d = info.to_dict()
        assert d["best_text"] == "test"
        assert d["confidence"] == 0.9
        assert len(d["candidates"]) == 1

    def test_ocr_info_from_dict(self):
        data = {
            "best_text": "test",
            "confidence": 0.9,
            "candidates": [{"text": "test", "confidence": 0.9, "source": "1x"}],
        }
        info = OCRInfo.from_dict(data)
        assert info.best_text == "test"
        assert len(info.candidates) == 1

    def test_region_status_after_ocr_no_text(self):
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        regions = [
            TextRegion.create(
                polygon=[[10, 10], [200, 10], [200, 50], [10, 50]],
                bbox=[10, 10, 200, 50],
                is_tiny=True,
            )
        ]
        result = run_ocr(img, regions, ocr_engine=None)
        assert result[0].status in ("detected", "needs_manual_input")
