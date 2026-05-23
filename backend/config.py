import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    api_base: str = ""
    api_key: str = ""
    timeout: int = 60
    max_retries: int = 3
    max_workers: int = 4
    temperature: float = 0.0
    top_p: float = 0.1
    response_format: str = "json"


@dataclass
class OCRConfig:
    provider: str = "paddle"
    api_url: str = ""
    token: str = ""
    language: str = "zh-CN"
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_chart_recognition: bool = False


@dataclass
class InpaintConfig:
    provider: str = "opencv"
    method: str = "telea"
    radius: int = 3


@dataclass
class RenderConfig:
    default_font: str = "NotoSansCJKsc-Regular.otf"
    min_font_size: float = 8.0
    max_font_size: float = 200.0
    font_dirs: list = field(default_factory=lambda: [])


@dataclass
class StorageConfig:
    data_dir: Path = field(default_factory=lambda: Path("data/projects"))
    max_file_size_mb: int = 50
    max_image_pixels: int = 40_000_000
    max_image_width: int = 12000
    max_image_height: int = 12000
    allowed_mime_types: tuple = (
        "image/png", "image/jpeg", "image/webp", "image/tiff"
    )


@dataclass
class SecurityConfig:
    require_auth: bool = False
    api_token: str = ""
    allowed_origins: list[str] = field(default_factory=lambda: [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ])


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False


def _try_load_apikey_file() -> str:
    candidates = [
        Path(__file__).parent.parent.parent / "apikey.txt",
        Path(__file__).parent.parent / "apikey.txt",
        Path("apikey.txt"),
        Path.home() / "apikey.txt",
    ]
    for p in candidates:
        if p.exists():
            try:
                key = p.read_text(encoding="utf-8").strip()
                if key and key.startswith("sk-"):
                    return key
            except Exception:
                pass
    return ""


def load_config() -> AppConfig:
    config = AppConfig()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        api_key = _try_load_apikey_file()
    config.llm.api_key = api_key

    config.llm.api_base = os.environ.get(
        "DEEPSEEK_API_BASE", "https://api.deepseek.com"
    )
    config.llm.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

    env_model = os.environ.get("TEXTPATCH_LLM_MODEL")
    if env_model:
        config.llm.model = env_model

    env_ocr_url = os.environ.get("TEXTPATCH_OCR_API_URL")
    if env_ocr_url:
        config.ocr.api_url = env_ocr_url

    env_ocr_token = os.environ.get("TEXTPATCH_OCR_TOKEN")
    if env_ocr_token:
        config.ocr.token = env_ocr_token

    if not config.ocr.api_url and not config.ocr.token:
        config.ocr.provider = "rapid"
        config.ocr.api_url = ""
        config.ocr.token = ""

    env_data_dir = os.environ.get("TEXTPATCH_DATA_DIR")
    if env_data_dir:
        config.storage.data_dir = Path(env_data_dir)

    env_debug = os.environ.get("TEXTPATCH_DEBUG", "")
    if env_debug.lower() in ("1", "true", "yes"):
        config.debug = True

    # Security config
    config.security.require_auth = os.environ.get(
        "TEXTPATCH_REQUIRE_AUTH", "false"
    ).lower() in ("1", "true", "yes")
    config.security.api_token = os.environ.get("TEXTPATCH_API_TOKEN", "")
    env_origins = os.environ.get("TEXTPATCH_ALLOWED_ORIGINS", "")
    if env_origins:
        config.security.allowed_origins = [
            o.strip() for o in env_origins.split(",") if o.strip()
        ]

    config.host = os.environ.get("TEXTPATCH_HOST", "127.0.0.1")
    env_port = os.environ.get("TEXTPATCH_PORT", "")
    if env_port:
        config.port = int(env_port)

    return config


app_config = load_config()
