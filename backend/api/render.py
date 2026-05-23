from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.core.pipeline import Pipeline
from backend.exceptions import ProjectNotFoundError, ImageNotFoundError, NoFontAvailableError

router = APIRouter(prefix="/api/projects", tags=["render"])


def get_pipeline() -> Pipeline:
    from backend.config import app_config
    return Pipeline(app_config)


@router.post("/{project_id}/render")
async def render(
    project_id: str,
    pipeline: Pipeline = Depends(get_pipeline),
):
    try:
        project = pipeline.render(project_id)
        return {
            "project_id": project.id,
            "final_image_path": project.final_image_path,
            "regions": [r.to_dict() for r in project.regions],
        }
    except ValueError:
        raise ProjectNotFoundError(project_id)
    except FileNotFoundError:
        raise ImageNotFoundError(project_id)
