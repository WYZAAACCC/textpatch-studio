from __future__ import annotations
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from backend.core.pipeline import Pipeline
from backend.storage.file_store import FileStore
from backend.exceptions import ProjectNotFoundError

router = APIRouter(prefix="/api/projects", tags=["projects"])


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


@router.post("", response_model=ProjectResponse)
async def create_project(
    name: str = "Untitled",
    file: UploadFile = File(...),
    pipeline: Pipeline = Depends(get_pipeline),
):
    if not FileStore.is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.filename}",
        )

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        project = pipeline.create_project(name, tmp_path)
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
    x: int
    y: int
    width: int
    height: int


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
    x: int
    y: int
    width: int
    height: int
    mode: str = "auto"  # "auto" or "simple"


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
    region_ids: list
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
