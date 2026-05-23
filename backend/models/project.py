from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Project:
    id: str
    name: str
    original_image_path: str
    clean_base_path: Optional[str]
    final_image_path: Optional[str]
    width: int
    height: int
    regions: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @staticmethod
    def create(name: str, image_path: str, width: int, height: int) -> Project:
        ts = datetime.now().strftime("%Y%m%d")
        uid = uuid.uuid4().hex[:8]
        return Project(
            id=f"p_{ts}_{uid}",
            name=name,
            original_image_path=image_path,
            clean_base_path=None,
            final_image_path=None,
            width=width,
            height=height,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "original_image_path": self.original_image_path,
            "clean_base_path": self.clean_base_path,
            "final_image_path": self.final_image_path,
            "width": self.width,
            "height": self.height,
            "regions": [r.to_dict() if hasattr(r, "to_dict") else r for r in self.regions],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Project:
        from backend.models.region import TextRegion
        regions_data = data.get("regions", [])
        regions = [TextRegion.from_dict(r) if isinstance(r, dict) else r for r in regions_data]
        return cls(
            id=data["id"],
            name=data["name"],
            original_image_path=data["original_image_path"],
            clean_base_path=data.get("clean_base_path"),
            final_image_path=data.get("final_image_path"),
            width=data["width"],
            height=data["height"],
            regions=regions,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
