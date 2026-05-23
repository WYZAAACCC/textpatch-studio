from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class OCRDetectionResult:
    text: str
    confidence: float
    bbox: list
    polygon: list = None
    angle: float = 0.0


@dataclass
class OCRRecognizeResult:
    text: str
    confidence: float
    source: str


class OCREngine(ABC):
    @abstractmethod
    def detect(self, image_bytes: bytes, file_type: int = 1) -> list[OCRDetectionResult]:
        raise NotImplementedError

    @abstractmethod
    def recognize(self, image_bytes: bytes, file_type: int = 1) -> list[OCRRecognizeResult]:
        raise NotImplementedError

    @abstractmethod
    def detect_and_recognize(
        self, image_bytes: bytes, file_type: int = 1
    ) -> list[OCRDetectionResult]:
        raise NotImplementedError
