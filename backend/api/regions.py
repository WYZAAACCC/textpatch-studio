from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional

from backend.core.pipeline import Pipeline
from backend.dependencies import get_pipeline
from backend.security import require_api_token
from backend.exceptions import ProjectNotFoundError

router = APIRouter(
    prefix="/api/projects",
    tags=["regions"],
    dependencies=[Depends(require_api_token)],
)

ALLOWED_STYLE_KEYS = {
    "font_family", "font_size", "color", "stroke_color",
    "bold", "italic", "underline", "letter_spacing",
    "line_height", "align", "v_align",
}


class RegionUpdateRequest(BaseModel):
    final_text: Optional[str] = Field(None, max_length=2000)
    style: Optional[dict] = None
    status: Optional[str] = None



@router.get("/{project_id}/regions")
async def list_regions(
    project_id: str,
    pipeline: Pipeline = Depends(get_pipeline),
):
    project = pipeline.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    return {"regions": [r.to_dict() for r in project.regions]}


@router.patch("/{project_id}/regions/{region_id}")
async def update_region(
    project_id: str,
    region_id: str,
    request: RegionUpdateRequest,
    pipeline: Pipeline = Depends(get_pipeline),
):
    style = request.style
    if style is not None:
        style = {k: v for k, v in style.items() if k in ALLOWED_STYLE_KEYS}

    project = pipeline.update_region(
        project_id,
        region_id,
        final_text=request.final_text,
        style=style,
        status=request.status,
    )
    if not project:
        raise ProjectNotFoundError(project_id)

    for region in project.regions:
        if region.id == region_id:
            return region.to_dict()

    raise ProjectNotFoundError(project_id)
