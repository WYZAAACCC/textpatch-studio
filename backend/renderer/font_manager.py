from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FontManager:
    def __init__(self, font_dirs: list = None):
        self.font_dirs = []
        self._font_map: dict[str, str] = {}

        default_dirs = [
            Path("fonts"),
            Path(__file__).parent.parent.parent / "fonts",
        ]

        if font_dirs:
            for d in font_dirs:
                self.font_dirs.append(Path(d))

        for d in default_dirs:
            self.font_dirs.append(d)

        system_font_dirs = self._get_system_font_dirs()
        self.font_dirs.extend(system_font_dirs)

        self._scan_fonts()

        if not self._font_map:
            self._try_ensure_fonts_dir()

    def _get_system_font_dirs(self) -> list[Path]:
        dirs = []
        if os.name == "nt":
            windir = os.environ.get("WINDIR", r"C:\Windows")
            dirs.append(Path(windir) / "Fonts")
            local_app = os.environ.get("LOCALAPPDATA", "")
            if local_app:
                dirs.append(Path(local_app) / "Microsoft" / "Windows" / "Fonts")
        elif os.name == "posix":
            dirs.extend([
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path.home() / ".fonts",
                Path.home() / ".local" / "share" / "fonts",
            ])
        return dirs

    def _try_ensure_fonts_dir(self):
        """Ensure fonts directory exists and has at least a .gitkeep to prompt user."""
        project_fonts = Path(__file__).parent.parent.parent / "fonts"
        project_fonts.mkdir(parents=True, exist_ok=True)

    def get_fonts_dir(self) -> Path:
        """Return the project fonts directory path."""
        return (Path(__file__).parent.parent.parent / "fonts").resolve()

    def download_font(self, font_name: str = "NotoSansCJKsc-Regular.otf") -> bool:
        """Attempt to download a Chinese font. Returns True on success."""
        try:
            from backend.renderer.font_downloader import download_chinese_font
            font_path = download_chinese_font(
                target_dir=self.get_fonts_dir()
            )
            if font_path and font_path.exists():
                self._scan_fonts()
                return True
        except Exception as e:
            logger.error("Font download failed: %s", e)
        return False

    def _scan_fonts(self):
        self._font_map.clear()
        for font_dir in self.font_dirs:
            if not font_dir.exists():
                continue
            for root, _, files in os.walk(str(font_dir)):
                for f in files:
                    fpath = Path(root) / f
                    if fpath.suffix.lower() in (".otf", ".ttf", ".ttc"):
                        name = fpath.stem.lower()
                        self._font_map[name] = str(fpath)

                        if "noto" in name and "cjk" in name:
                            cjk_aliases = [
                                "noto sans cjk sc",
                                "noto sans cjk",
                                "noto sans",
                                "思源黑体",
                            ]
                            for alias in cjk_aliases:
                                if alias not in self._font_map:
                                    self._font_map[alias] = str(fpath)

                        if "source" in name and "han" in name:
                            han_aliases = [
                                "source han sans",
                                "source han sans sc",
                                "思源黑体",
                            ]
                            for alias in han_aliases:
                                if alias not in self._font_map:
                                    self._font_map[alias] = str(fpath)

                        if name in ("simhei", "msyh", "msjh", "microsoftyahei", "microsoftjhenghei"):
                            sans_aliases = [
                                "noto sans cjk sc",
                                "noto sans cjk",
                                "noto sans",
                            ]
                            for alias in sans_aliases:
                                if alias not in self._font_map:
                                    self._font_map[alias] = str(fpath)

    def get_font_path(self, font_family: str) -> Optional[str]:
        key = font_family.lower()
        if key in self._font_map:
            return self._font_map[key]

        for k, v in self._font_map.items():
            if key in k or k in key:
                return v

        return None

    def get_default_font_path(self) -> Optional[str]:
        defaults = [
            "notosanscjksc-regular",
            "noto sans cjk sc",
            "sourcehansanssc-regular",
            "source han sans sc",
            "simhei",
            "microsoftyahei",
            "msyh",
            "msjh",
            "sourcehansanssc-bold",
            "sourcehansanssc-light",
            "noto sans",
            "simsun",
            "microsoftjhenghei",
            "arial",
        ]

        for name in defaults:
            path = self.get_font_path(name)
            if path:
                return path

        if self._font_map:
            return next(iter(self._font_map.values()))

        return None

    def has_any_font(self) -> bool:
        return len(self._font_map) > 0

    def list_available_fonts(self) -> list[dict]:
        fonts = []
        seen = set()
        for name, path in self._font_map.items():
            if path not in seen:
                seen.add(path)
                fonts.append({"name": name, "path": path})
        return fonts
