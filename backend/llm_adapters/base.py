from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextCorrectionRequest:
    ocr_best: str
    ocr_candidates: list
    neighbor_texts: list
    risk_flags: list
    scene_hint: str = ""


@dataclass
class TextCorrectionResponse:
    corrected_text: str
    confidence: float
    correction_type: str
    changed_chars: list
    uncertain_chars: list
    needs_human: bool
    raw_response: dict
    is_formula: bool = False
    latex: str = ""


class LLMClient(ABC):
    @abstractmethod
    def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        raise NotImplementedError
