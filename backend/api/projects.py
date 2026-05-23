from __future__ import annotations
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, Field

from backend.core.pipeline import Pipeline
from backend.security import require_api_token
from backend.storage.file_store import FileStore
from backend.exceptions import ProjectNotFoundError

router = APIRouter(
    prefix="/api/projects",
    tags=["projects"],
    dependencies=[Depends(require_api_token)],
)


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    width: int
    height: int


class ProjectListResponse(BaseModel):
    projects: list


def get_pipeline() -> Pipeline:
    from backend.config import app_config
    return Pipeline(app_config)


async def _save_upload_chunked(file: UploadFile, max_bytes: int) -> Path:
    import tempfile
    suffix = Path(file.filename or "").suffix.lower()
    total = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds maximum size of {max_bytes // (1024*1024)} MB",
                )
            tmp.write(chunk)
    return tmp_path


@router.post("", response_model=ProjectResponse)
async def create_project(
    name: str = "Untitled",
    file: UploadFile = File(...),
    pipeline: Pipeline = Depends(get_pipeline),
):
    from backend.config import app_config

    if not FileStore.is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.filename}",
        )

    max_bytes = app_config.storage.max_file_size_mb * 1024 * 1024
    tmp_path = await _save_upload_chunked(file, max_bytes)

    try:
        from backend.core.image_validation import validate_image_file, InvalidImageError
        validate_image_file(tmp_path, app_config.storage)
        project = pipeline.create_project(name, tmp_path)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return ProjectResponse(
        project_id=project.id,
        name=project.name,
        width=project.width,
        height=project.height,
    )


@router.get("")
async def list_projects(pipeline: Pipeline = Depends(get_pipeline)):
    return {"projects": pipeline.list_projects()}


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    pipeline: Pipeline = Depends(get_pipeline),
):
    project = pipeline.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    return project.to_dict()


class RestoreRequest(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., ge=1, le=12000)
    height: int = Field(..., ge=1, le=12000)


@router.post("/{project_id}/restore")
async def restore_region(
    project_id: str,
    body: RestoreRequest,
    pipeline: Pipeline = Depends(get_pipeline),
):
    try:
        project = pipeline.restore_region(
            project_id, body.x, body.y, body.width, body.height
        )
        return project.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


class DetectRegionRequest(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., ge=1, le=12000)
    height: int = Field(..., ge=1, le=12000)
    mode: str = "auto"


@router.post("/{project_id}/detect-region")
async def detect_region(
    project_id: str,
    body: DetectRegionRequest,
    pipeline: Pipeline = Depends(get_pipeline),
):
    try:
        simple = body.mode == "simple"
        project = pipeline.detect_region(
            project_id, body.x, body.y, body.width, body.height,
            simple_mode=simple,
        )
        return project.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


class BatchUpdateRequest(BaseModel):
    region_ids: list[str] = Field(..., min_length=1, max_length=500)
    style: Optional[dict] = None
    status: Optional[str] = None


@router.post("/{project_id}/batch-update-regions")
async def batch_update_regions(
    project_id: str,
    body: BatchUpdateRequest,
    pipeline: Pipeline = Depends(get_pipeline),
):
    project = pipeline.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)

    for region in project.regions:
        if region.id in body.region_ids:
            if body.style:
                if region.style is None:
                    from backend.models.style import TextStyle
                    region.style = TextStyle()
                for key, value in body.style.items():
                    if hasattr(region.style, key):
                        if key in ("color", "stroke_color") and isinstance(value, list):
                            value = tuple(value)
                        setattr(region.style, key, value)
            if body.status:
                region.status = body.status

    pipeline.project_store.save(project)
    return project.to_dict()
