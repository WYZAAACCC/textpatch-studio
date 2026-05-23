from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class RenderInfo:
    transform: str = "none"
    blend_mode: str = "normal"
    overflow: bool = False
    rendered_layer_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "transform": self.transform,
            "blend_mode": self.blend_mode,
            "overflow": self.overflow,
            "rendered_layer_path": self.rendered_layer_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RenderInfo:
        if data is None:
            return cls()
        return cls(
            transform=data.get("transform", "none"),
            blend_mode=data.get("blend_mode", "normal"),
            overflow=data.get("overflow", False),
            rendered_layer_path=data.get("rendered_layer_path"),
        )
