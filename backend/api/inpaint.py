from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from backend.core.pipeline import Pipeline
from backend.security import require_api_token
from backend.exceptions import ProjectNotFoundError, ImageNotFoundError, ImageReadError

router = APIRouter(
    prefix="/api/projects",
    tags=["inpaint"],
    dependencies=[Depends(require_api_token)],
)


class InpaintRequest(BaseModel):
    region_ids: Optional[list] = None
    method: str = "auto"


def get_pipeline() -> Pipeline:
    from backend.config import app_config
    return Pipeline(app_config)


@router.post("/{project_id}/inpaint")
async def inpaint(
    project_id: str,
    request: InpaintRequest = InpaintRequest(),
    pipeline: Pipeline = Depends(get_pipeline),
):
    try:
        simple_mode = request.method == "simple"
        project = pipeline.inpaint(project_id, region_ids=request.region_ids,
                                    simple_mode=simple_mode)
        return {
            "project_id": project.id,
            "clean_base_path": project.clean_base_path,
            "regions": [r.to_dict() for r in project.regions],
        }
    except ValueError:
        raise ProjectNotFoundError(project_id)
    except FileNotFoundError:
        raise ImageNotFoundError(project_id)
