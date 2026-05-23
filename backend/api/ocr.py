from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.core.pipeline import Pipeline
from backend.dependencies import get_pipeline
from backend.security import require_api_token
from backend.exceptions import (
    ProjectNotFoundError,
    ImageNotFoundError,
    ImageReadError,
    NoOCREngineError,
)

router = APIRouter(
    prefix="/api/projects",
    tags=["ocr"],
    dependencies=[Depends(require_api_token)],
)


class DetectRequest(BaseModel):
    detect_small_text: bool = True
    scales: list[float] = Field(default_factory=lambda: [1, 2, 3])
    language: str = Field("zh-CN", max_length=16)



@router.post("/{project_id}/detect")
async def detect_text(
    project_id: str,
    request: DetectRequest = DetectRequest(),
    pipeline: Pipeline = Depends(get_pipeline),
):
    try:
        project = pipeline.detect(
            project_id,
            detect_small_text=request.detect_small_text,
            scales=request.scales,
            language=request.language,
        )
        return {
            "project_id": project.id,
            "region_count": len(project.regions),
            "regions": [r.to_dict() for r in project.regions],
        }
    except ValueError:
        raise ProjectNotFoundError(project_id)
    except FileNotFoundError:
        raise ImageNotFoundError(project_id)
    except RuntimeError as e:
        if "OCR" in str(e) or "ocr" in str(e):
            raise NoOCREngineError()
        raise


@router.post("/{project_id}/ocr")
async def run_ocr(
    project_id: str,
    pipeline: Pipeline = Depends(get_pipeline),
):
    try:
        project = pipeline.ocr(project_id)
        return {
            "project_id": project.id,
            "regions": [r.to_dict() for r in project.regions],
        }
    except ValueError:
        raise ProjectNotFoundError(project_id)
    except FileNotFoundError:
        raise ImageNotFoundError(project_id)
