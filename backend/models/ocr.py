from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class OCRCandidate:
    text: str
    confidence: float
    source: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OCRCandidate:
        return cls(
            text=data["text"],
            confidence=data.get("confidence", 0.0),
            source=data.get("source", ""),
        )


@dataclass
class OCRInfo:
    best_text: str = ""
    confidence: float = 0.0
    candidates: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_text": self.best_text,
            "confidence": self.confidence,
            "candidates": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: dict) -> OCRInfo:
        if data is None:
            return cls()
        candidates = [
            OCRCandidate.from_dict(c) if isinstance(c, dict) else c
            for c in data.get("candidates", [])
        ]
        return cls(
            best_text=data.get("best_text", ""),
            confidence=data.get("confidence", 0.0),
            candidates=candidates,
        )
