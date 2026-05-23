from __future__ import annotations
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import app_config
from backend.api import projects, regions, ocr, correction, inpaint, render, export
from backend.exceptions import register_exception_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_health_status() -> dict:
    """Return comprehensive system health status with diagnostics."""
    from backend.renderer.font_manager import FontManager
    from backend.ocr_adapters.rapid import RapidOCREngine

    fm = FontManager()
    fonts_available = fm.has_any_font()
    fonts_count = len(fm.list_available_fonts())

    local_ocr_available = RapidOCREngine.is_available()

    llm_configured = bool(app_config.llm.api_key)
    paddle_configured = bool(app_config.ocr.api_url and app_config.ocr.token)

    issues = []
    if not fonts_available:
        issues.append({
            "severity": "warning",
            "component": "fonts",
            "message": "No Chinese fonts detected. Text rendering will fail.",
            "action": "Call POST /api/fonts/download or place fonts in the fonts/ directory.",
        })
    if not local_ocr_available and not paddle_configured:
        issues.append({
            "severity": "error",
            "component": "ocr",
            "message": "No OCR engine available. Text detection will fail.",
            "action": "Install rapidocr-onnxruntime or configure PaddleOCR API credentials.",
        })
    if not llm_configured:
        issues.append({
            "severity": "info",
            "component": "llm",
            "message": "No LLM API key configured. Using mock client (no text correction).",
            "action": "Set DEEPSEEK_API_KEY environment variable or create apikey.txt.",
        })

    font_download_enabled = __import__("os").environ.get(
        "TEXTPATCH_ENABLE_FONT_DOWNLOAD", "false"
    ).lower() in ("1", "true", "yes")

    status = "ok" if not issues or all(i["severity"] != "error" for i in issues) else "degraded"

    return {
        "status": status,
        "version": "0.1.0",
        "components": {
            "fonts": {
                "available": fonts_available,
                "count": fonts_count,
                "download_enabled": font_download_enabled,
            },
            "ocr": {
                "local_available": local_ocr_available,
                "paddle_api_configured": paddle_configured,
                "provider": app_config.ocr.provider,
            },
            "llm": {
                "configured": llm_configured,
                "provider": app_config.llm.provider,
                "mock_in_use": not llm_configured,
            },
            "inpaint": {
                "provider": app_config.inpaint.provider,
                "method": app_config.inpaint.method,
                "lama_real": False,
            },
        },
        "issues": issues,
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="TextPatch Studio",
        description="AI生成图中文字后处理工具 - 检测乱码小字、OCR/LLM校正、擦除伪字、真实字体重新排版",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.security.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-TextPatch-Token"],
    )

    app.include_router(projects.router)
    app.include_router(regions.router)
    app.include_router(ocr.router)
    app.include_router(correction.router)
    app.include_router(inpaint.router)
    app.include_router(render.router)
    app.include_router(export.router)

    register_exception_handlers(app)

    data_dir = app_config.storage.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/health")
    async def health():
        return _get_health_status()

    @app.get("/api/projects/{project_id}/image/{image_type}")
    async def get_project_image(project_id: str, image_type: str):
        """Serve project images (original, clean_base, final, preview)."""
        import mimetypes
        from backend.core.pipeline import Pipeline
        from fastapi.responses import FileResponse
        pl = Pipeline(app_config)
        project = pl.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if image_type not in {"original", "clean_base", "final", "preview"}:
            raise HTTPException(status_code=404, detail=f"Unknown image type: {image_type}")

        path = None
        if image_type == "original":
            path = project.original_image_path
        elif image_type == "clean_base":
            path = project.clean_base_path
        elif image_type == "final":
            path = project.final_image_path
        elif image_type == "preview":
            from backend.storage.file_store import FileStore
            fs = FileStore(app_config.storage.data_dir)
            path = fs.get_preview(project_id)

        if not path or not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"Image {image_type} not available")
        media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.get("/api/fonts")
    async def list_fonts():
        from backend.renderer.font_manager import FontManager
        fm = FontManager(app_config.render.font_dirs)
        return {"fonts": fm.list_available_fonts()}

    @app.post("/api/fonts/download")
    async def download_font():
        from backend.renderer.font_manager import FontManager
        from backend.exceptions import FontDownloadError as FontDownloadExc

        fm = FontManager(app_config.render.font_dirs)
        if fm.download_font():
            return {
                "status": "ok",
                "message": "Font downloaded successfully.",
                "fonts": fm.list_available_fonts(),
            }
        raise FontDownloadExc("Failed to download font. Check server logs for details.")

    # Static frontend files — mounted last so API routes take priority
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug,
    )
