"""Custom exceptions and error handling for TextPatch Studio.

Provides user-friendly error messages with actionable guidance.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class TextPatchError(Exception):
    """Base exception for all TextPatch Studio errors."""
    status_code: int = 500
    detail: str = "An unexpected error occurred."
    guidance: str = ""

    def to_dict(self) -> dict:
        result = {
            "error": self.__class__.__name__,
            "detail": self.detail,
        }
        if self.guidance:
            result["guidance"] = self.guidance
        return result


class ProjectNotFoundError(TextPatchError):
    status_code = 404

    def __init__(self, project_id: str):
        self.detail = f"Project '{project_id}' not found."
        self.guidance = (
            "The project may have been deleted or the ID is incorrect. "
            "Use GET /api/projects to list all projects."
        )


class ImageNotFoundError(TextPatchError):
    status_code = 404

    def __init__(self, project_id: str):
        self.detail = f"Original image not found for project '{project_id}'."
        self.guidance = (
            "The original image file is missing from storage. "
            "Re-create the project with the original image."
        )


class ImageReadError(TextPatchError):
    status_code = 400

    def __init__(self, path: str):
        filename = Path(path).name if path else "unknown"
        self.detail = f"Failed to read image file: {filename}"
        self.guidance = (
            "The image file may be corrupted or in an unsupported format. "
            "Supported formats: PNG, JPG, JPEG, BMP, TIFF."
        )


class NoOCREngineError(TextPatchError):
    status_code = 503

    def __init__(self):
        self.detail = "No OCR engine is available."
        self.guidance = (
            "Options to enable OCR:\n"
            "  1. Install local OCR: pip install rapidocr-onnxruntime\n"
            "  2. Configure PaddleOCR API: set TEXTPATCH_OCR_API_URL and "
            "TEXTPATCH_OCR_TOKEN environment variables"
        )


class NoFontAvailableError(TextPatchError):
    status_code = 503

    def __init__(self):
        self.detail = "No fonts are available for text rendering."
        self.guidance = (
            "Solutions:\n"
            "  1. Install Chinese fonts on your system\n"
            "  2. Call POST /api/fonts/download to auto-download Noto Sans CJK SC\n"
            "  3. Place any .otf/.ttf font file in the fonts/ directory\n"
            "  4. On Windows, install fonts via: Settings > Personalization > Fonts"
        )


class NoLLMAPIKeyError(TextPatchError):
    status_code = 503

    def __init__(self):
        self.detail = "LLM API key is not configured. Text correction is unavailable."
        self.guidance = (
            "Set your DeepSeek API key via:\n"
            "  - Environment variable: DEEPSEEK_API_KEY=sk-...\n"
            "  - File: place apikey.txt in the project root or home directory\n"
            "Without an API key, the mock LLM client will be used (no actual correction)."
        )


class FontDownloadError(TextPatchError):
    status_code = 500

    def __init__(self, detail: str = ""):
        self.detail = detail or "Failed to download font."
        self.guidance = (
            "Network issues or the download source may be unavailable.\n"
            "Manual download:\n"
            "  1. Visit: https://github.com/googlefonts/noto-cjk/releases\n"
            "  2. Download: 03_NotoSansCJKsc.zip\n"
            "  3. Extract NotoSansCJKsc-Regular.otf to the fonts/ directory"
        )


class PipelineError(TextPatchError):
    status_code = 422

    def __init__(self, stage: str, detail: str):
        self.detail = f"Pipeline error at '{stage}' stage: {detail}"
        self.guidance = "Check the server logs for more details."


async def textpatch_exception_handler(request: Request, exc: TextPatchError):
    """Global exception handler for TextPatch errors."""
    logger.error(
        "%s (status=%d): %s",
        exc.__class__.__name__, exc.status_code, exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Check the server logs.",
            "guidance": (
                "If this persists:\n"
                "  1. Check the server console for full error details\n"
                "  2. Verify all dependencies are installed: pip install -e .\n"
                "  3. Check that data directories are writable"
            ),
        },
    )


def register_exception_handlers(app):
    """Register all custom exception handlers on the FastAPI app."""
    from fastapi.exceptions import RequestValidationError

    app.add_exception_handler(TextPatchError, textpatch_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("Value error (422): %s", str(exc)[:200])
        return JSONResponse(
            status_code=422,
            content={
                "error": "UnprocessableEntity",
                "detail": str(exc)[:500],
                "guidance": "Check that all IDs and parameter values are valid.",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "detail": "Invalid request parameters.",
                "validation_errors": exc.errors(),
                "guidance": "Check the request body or query parameters match the expected format.",
            },
        )
