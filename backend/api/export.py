from __future__ import annotations
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from backend.core.pipeline import Pipeline
from backend.security import require_api_token
from backend.exceptions import ProjectNotFoundError, ImageNotFoundError

router = APIRouter(
    prefix="/api/projects",
    tags=["export"],
    dependencies=[Depends(require_api_token)],
)


class ExportRequest(BaseModel):
    format: Literal["png", "jpeg", "jpg", "webp", "zip"] = "png"
    quality: int = Field(95, ge=1, le=100)
    scale: float = Field(1.0, ge=0.1, le=4.0)
    include_project_file: bool = False


def get_pipeline() -> Pipeline:
    from backend.config import app_config
    return Pipeline(app_config)


@router.post("/{project_id}/export")
async def export_project(
    project_id: str,
    request: ExportRequest = ExportRequest(),
    pipeline: Pipeline = Depends(get_pipeline),
):
    try:
        path = pipeline.export_project(
            project_id,
            format=request.format,
            quality=request.quality,
            scale=request.scale,
            include_project_file=request.include_project_file,
        )
        media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return FileResponse(
            str(path),
            media_type=media_type,
            filename=Path(path).name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise ImageNotFoundError(project_id)
