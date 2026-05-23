from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class ReviewInfo:
    status: str = "pending"
    reviewer: str = ""
    comment: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reviewer": self.reviewer,
            "comment": self.comment,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReviewInfo:
        if data is None:
            return cls()
        return cls(
            status=data.get("status", "pending"),
            reviewer=data.get("reviewer", ""),
            comment=data.get("comment", ""),
            reviewed_at=data.get("reviewed_at", ""),
        )


@dataclass
class TextRegion:
    id: str
    polygon: list = field(default_factory=list)
    bbox: list = field(default_factory=list)
    angle: float = 0.0
    source: str = "detection"
    confidence: float = 0.0
    is_tiny: bool = False
    status: str = "detected"
    ocr: Optional[Any] = None
    llm: Optional[Any] = None
    final_text: str = ""
    risk_flags: list = field(default_factory=list)
    review: ReviewInfo = field(default_factory=ReviewInfo)
    style: Optional[Any] = None
    render: Optional[Any] = None
    is_formula: bool = False
    latex_source: str = ""

    @staticmethod
    def create(
        polygon: list,
        bbox: list,
        angle: float = 0.0,
        source: str = "detection",
        confidence: float = 0.0,
        is_tiny: bool = False,
    ) -> TextRegion:
        uid = uuid.uuid4().hex[:6]
        return TextRegion(
            id=f"region_{uid}",
            polygon=polygon,
            bbox=bbox,
            angle=angle,
            source=source,
            confidence=confidence,
            is_tiny=is_tiny,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "polygon": self.polygon,
            "bbox": self.bbox,
            "angle": self.angle,
            "source": self.source,
            "confidence": self.confidence,
            "is_tiny": self.is_tiny,
            "status": self.status,
            "ocr": self.ocr.to_dict() if hasattr(self.ocr, "to_dict") else self.ocr,
            "llm": self.llm.to_dict() if hasattr(self.llm, "to_dict") else self.llm,
            "final_text": self.final_text,
            "risk_flags": self.risk_flags,
            "review": self.review.to_dict() if hasattr(self.review, "to_dict") else self.review,
            "style": self.style.to_dict() if hasattr(self.style, "to_dict") else self.style,
            "render": self.render.to_dict() if hasattr(self.render, "to_dict") else self.render,
            "is_formula": self.is_formula,
            "latex_source": self.latex_source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TextRegion:
        from backend.models.ocr import OCRInfo
        from backend.models.llm import LLMCorrectionInfo
        from backend.models.style import TextStyle
        from backend.models.render import RenderInfo

        ocr = None
        if data.get("ocr") and isinstance(data["ocr"], dict):
            ocr = OCRInfo.from_dict(data["ocr"])

        llm = None
        if data.get("llm") and isinstance(data["llm"], dict):
            llm = LLMCorrectionInfo.from_dict(data["llm"])

        style = None
        if data.get("style") and isinstance(data["style"], dict):
            style = TextStyle.from_dict(data["style"])

        render = None
        if data.get("render") and isinstance(data["render"], dict):
            render = RenderInfo.from_dict(data["render"])

        review = ReviewInfo.from_dict(data.get("review"))

        return cls(
            id=data["id"],
            polygon=data.get("polygon", []),
            bbox=data.get("bbox", []),
            angle=data.get("angle", 0.0),
            source=data.get("source", "detection"),
            confidence=data.get("confidence", 0.0),
            is_tiny=data.get("is_tiny", False),
            status=data.get("status", "detected"),
            ocr=ocr,
            llm=llm,
            final_text=data.get("final_text", ""),
            risk_flags=data.get("risk_flags", []),
            review=review,
            style=style,
            render=render,
            is_formula=data.get("is_formula", False),
            latex_source=data.get("latex_source", ""),
        )
