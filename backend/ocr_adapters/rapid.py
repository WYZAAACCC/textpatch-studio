"""Local OCR engine using RapidOCR (ONNX Runtime).

Provides offline OCR capabilities without requiring an external API.
Uses rapidocr-onnxruntime when available, with graceful fallback.
"""

from __future__ import annotations
import logging
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image

from backend.ocr_adapters.base import OCREngine, OCRDetectionResult

logger = logging.getLogger(__name__)

_RAPID_OCR_AVAILABLE = False
_RAPID_OCR_ERROR = None

try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    _RAPID_OCR_AVAILABLE = True
except ImportError as e:
    _RAPID_OCR_ERROR = str(e)
    logger.debug("rapidocr-onnxruntime not available: %s", e)


class RapidOCREngine(OCREngine):
    """Local OCR engine backed by RapidOCR (ONNX Runtime).

    Falls back to PaddleOCR HTTP API if rapidocr-onnxruntime is not installed
    and Paddle API credentials are configured.
    """

    def __init__(self, **kwargs):
        self._engine = None
        self._fallback_engine = None
        self._language = kwargs.get("language", "ch_PP-OCRv4")

    def _get_engine(self):
        """Lazy-initialize the RapidOCR engine."""
        if self._engine is not None:
            return self._engine

        if not _RAPID_OCR_AVAILABLE:
            raise RuntimeError(
                "Local OCR is not available. Install rapidocr-onnxruntime:\n"
                "  pip install rapidocr-onnxruntime\n\n"
                "Original error: " + (_RAPID_OCR_ERROR or "unknown")
            )

        try:
            self._engine = _RapidOCR()
            logger.info("RapidOCR local engine initialized successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize RapidOCR: {e}\n"
                "Try reinstalling: pip install rapidocr-onnxruntime"
            ) from e

        return self._engine

    def _get_fallback_engine(self):
        """Try PaddleOCR HTTP API as fallback."""
        if self._fallback_engine is None:
            from backend.config import app_config
            from backend.ocr_adapters.paddle import PaddleOCREngine

            if not app_config.ocr.api_url or not app_config.ocr.token:
                raise RuntimeError(
                    "No OCR engine available.\n\n"
                    "Options:\n"
                    "1. Install local OCR: pip install rapidocr-onnxruntime\n"
                    "2. Configure PaddleOCR API: set TEXTPATCH_OCR_API_URL and "
                    "TEXTPATCH_OCR_TOKEN environment variables"
                )

            logger.info("Using PaddleOCR HTTP API as fallback")
            self._fallback_engine = PaddleOCREngine(
                api_url=app_config.ocr.api_url,
                token=app_config.ocr.token,
            )

        return self._fallback_engine

    def detect(self, image_bytes: bytes, file_type: int = 1) -> list[OCRDetectionResult]:
        if _RAPID_OCR_AVAILABLE:
            return self._detect_local(image_bytes)
        else:
            return self._get_fallback_engine().detect(image_bytes, file_type)

    def recognize(self, image_bytes: bytes, file_type: int = 1):
        if _RAPID_OCR_AVAILABLE:
            return self._recognize_local(image_bytes)
        else:
            return self._get_fallback_engine().recognize(image_bytes, file_type)

    def detect_and_recognize(
        self, image_bytes: bytes, file_type: int = 1
    ) -> list[OCRDetectionResult]:
        if _RAPID_OCR_AVAILABLE:
            return self._detect_and_recognize_local(image_bytes)
        else:
            return self._get_fallback_engine().detect_and_recognize(image_bytes, file_type)

    def _detect_and_recognize_local(
        self, image_bytes: bytes
    ) -> list[OCRDetectionResult]:
        engine = self._get_engine()

        image = _bytes_to_numpy(image_bytes)
        if image is None:
            return []

        try:
            result, elapse = engine(image)
        except Exception as e:
            logger.error("RapidOCR detection failed: %s", e)
            return []

        if result is None:
            return []

        logger.info(
            "RapidOCR: %d text regions detected in %.2fs",
            len(result), elapse
        )

        regions = []
        for item in result:
            box, text, score = item
            if not text or not text.strip():
                continue

            bbox = _box_four_to_rect(box)
            regions.append(OCRDetectionResult(
                text=text.strip(),
                confidence=float(score),
                bbox=bbox,
                polygon=box,
            ))

        return regions

    def _detect_local(self, image_bytes: bytes) -> list[OCRDetectionResult]:
        return self._detect_and_recognize_local(image_bytes)

    def _recognize_local(self, image_bytes: bytes):
        from backend.ocr_adapters.base import OCRRecognizeResult
        regions = self._detect_and_recognize_local(image_bytes)
        return [
            OCRRecognizeResult(
                text=r.text,
                confidence=r.confidence,
                source="rapidocr-local",
            )
            for r in regions
        ]

    @staticmethod
    def is_available() -> bool:
        """Check if local RapidOCR is installed and available."""
        return _RAPID_OCR_AVAILABLE

    @staticmethod
    def get_install_instructions() -> str:
        """Return installation instructions for local OCR."""
        if _RAPID_OCR_AVAILABLE:
            return "RapidOCR is already installed and available."

        return (
            "Local OCR requires rapidocr-onnxruntime.\n\n"
            "Install with:  pip install rapidocr-onnxruntime\n\n"
            "This package includes:\n"
            "  - ONNX Runtime (CPU)\n"
            "  - Pre-trained Chinese text detection & recognition models\n"
            "  - No GPU or CUDA required\n\n"
            "For GPU acceleration (optional):\n"
            "  pip install rapidocr-onnxruntime-gpu"
        )


def _bytes_to_numpy(image_bytes: bytes) -> Optional[np.ndarray]:
    """Convert image bytes to BGR numpy array."""
    try:
        import io
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)
    except Exception as e:
        logger.error("Failed to decode image bytes: %s", e)
        return None


def _box_four_to_rect(box) -> tuple:
    """Convert 4-point box [[x1,y1],...] to axis-aligned rect (x1,y1,x2,y2)."""
    if box is None or len(box) < 4:
        return (0, 0, 0, 0)
    pts = np.array(box, dtype=np.float32)
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )
