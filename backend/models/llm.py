from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChangedChar:
    from_char: str
    to_char: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "from_char": self.from_char,
            "to_char": self.to_char,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChangedChar:
        return cls(
            from_char=data.get("from_char", ""),
            to_char=data.get("to_char", ""),
            reason=data.get("reason", ""),
        )


@dataclass
class LLMCorrectionInfo:
    provider: str = ""
    model: str = ""
    suggested_text: str = ""
    confidence: float = 0.0
    correction_type: str = "unchanged"
    changed_chars: list = field(default_factory=list)
    needs_human: bool = True
    raw_response: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "suggested_text": self.suggested_text,
            "confidence": self.confidence,
            "correction_type": self.correction_type,
            "changed_chars": [
                c.to_dict() if hasattr(c, "to_dict") else c for c in self.changed_chars
            ],
            "needs_human": self.needs_human,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LLMCorrectionInfo:
        if data is None:
            return cls()
        changed_chars = [
            ChangedChar.from_dict(c) if isinstance(c, dict) else c
            for c in data.get("changed_chars", [])
        ]
        return cls(
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            suggested_text=data.get("suggested_text", ""),
            confidence=data.get("confidence", 0.0),
            correction_type=data.get("correction_type", "unchanged"),
            changed_chars=changed_chars,
            needs_human=data.get("needs_human", True),
            raw_response=data.get("raw_response", {}),
        )
