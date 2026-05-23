from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.pipeline import Pipeline
from backend.exceptions import ProjectNotFoundError, ImageNotFoundError

router = APIRouter(prefix="/api/projects", tags=["export"])


class ExportRequest(BaseModel):
    format: str = "png"
    quality: int = 100
    include_project_file: bool = True


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
            include_project_file=request.include_project_file,
        )
        return FileResponse(
            str(path),
            media_type="image/png",
            filename=path.name,
        )
    except ValueError:
        raise ProjectNotFoundError(project_id)
    except FileNotFoundError:
        raise ImageNotFoundError(project_id)
