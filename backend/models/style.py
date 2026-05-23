from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ShadowStyle:
    dx: float = 0.0
    dy: float = 0.0
    blur: float = 0.0
    color: tuple = (0, 0, 0, 128)

    def to_dict(self) -> dict:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "blur": self.blur,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ShadowStyle:
        if data is None:
            return cls()
        color = data.get("color", [0, 0, 0, 128])
        return cls(
            dx=data.get("dx", 0.0),
            dy=data.get("dy", 0.0),
            blur=data.get("blur", 0.0),
            color=tuple(color),
        )


@dataclass
class TextStyle:
    font_family: str = "Noto Sans CJK SC"
    font_path: Optional[str] = None
    font_size: float = 24.0
    font_weight: str = "normal"
    italic: bool = False
    underline: bool = False
    color: tuple = (0, 0, 0, 255)
    stroke_width: float = 0.0
    stroke_color: Optional[tuple] = None
    shadow: Optional[ShadowStyle] = None
    letter_spacing: float = 0.0
    line_height: float = 1.2
    align: str = "left"
    v_align: str = "center"
    v_offset_ratio: float = -1.0
    vertical: bool = False
    opacity: float = 1.0
    blend_mode: str = "normal"
    min_font_size: float = 8.0
    max_font_size: float = 200.0
    allow_wrap: bool = True
    allow_letter_spacing_adjust: bool = True
    allow_region_expand: bool = False

    def to_dict(self) -> dict:
        return {
            "font_family": self.font_family,
            "font_path": self.font_path,
            "font_size": self.font_size,
            "font_weight": self.font_weight,
            "italic": self.italic,
            "underline": self.underline,
            "color": list(self.color),
            "stroke_width": self.stroke_width,
            "stroke_color": list(self.stroke_color) if self.stroke_color else None,
            "shadow": self.shadow.to_dict() if self.shadow else None,
            "letter_spacing": self.letter_spacing,
            "line_height": self.line_height,
            "align": self.align,
            "v_align": self.v_align,
            "v_offset_ratio": self.v_offset_ratio,
            "vertical": self.vertical,
            "opacity": self.opacity,
            "blend_mode": self.blend_mode,
            "min_font_size": self.min_font_size,
            "max_font_size": self.max_font_size,
            "allow_wrap": self.allow_wrap,
            "allow_letter_spacing_adjust": self.allow_letter_spacing_adjust,
            "allow_region_expand": self.allow_region_expand,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TextStyle:
        if data is None:
            return cls()
        color = data.get("color", [0, 0, 0, 255])
        stroke_color = data.get("stroke_color")
        if stroke_color:
            stroke_color = tuple(stroke_color)
        shadow = None
        if data.get("shadow"):
            shadow = ShadowStyle.from_dict(data["shadow"])
        return cls(
            font_family=data.get("font_family", "Noto Sans CJK SC"),
            font_path=data.get("font_path"),
            font_size=data.get("font_size", 24.0),
            font_weight=data.get("font_weight", "normal"),
            italic=data.get("italic", False),
            underline=data.get("underline", False),
            color=tuple(color),
            stroke_width=data.get("stroke_width", 0.0),
            stroke_color=stroke_color,
            shadow=shadow,
            letter_spacing=data.get("letter_spacing", 0.0),
            line_height=data.get("line_height", 1.2),
            align=data.get("align", "left"),
            v_align=data.get("v_align", "center"),
            v_offset_ratio=data.get("v_offset_ratio", -1.0),
            vertical=data.get("vertical", False),
            opacity=data.get("opacity", 1.0),
            blend_mode=data.get("blend_mode", "normal"),
            min_font_size=data.get("min_font_size", 8.0),
            max_font_size=data.get("max_font_size", 200.0),
            allow_wrap=data.get("allow_wrap", True),
            allow_letter_spacing_adjust=data.get("allow_letter_spacing_adjust", True),
            allow_region_expand=data.get("allow_region_expand", False),
        )
