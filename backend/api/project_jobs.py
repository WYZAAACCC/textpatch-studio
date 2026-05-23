"""Project-scoped job submission endpoints for async pipeline stages."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.core.jobs import get_job_manager
from backend.dependencies import get_pipeline
from backend.security import require_api_token
from backend.exceptions import ProjectNotFoundError

router = APIRouter(
    prefix="/api/projects",
    tags=["project_jobs"],
    dependencies=[Depends(require_api_token)],
)


def _get_manager(pipeline=None):
    if pipeline is None:
        pipeline = get_pipeline()
    return get_job_manager(pipeline)


@router.post("/{project_id}/jobs/detect")
async def submit_detect_job(project_id: str, detect_small_text: bool = True):
    pipeline = get_pipeline()
    project = pipeline.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)

    jm = get_job_manager(pipeline)

    regions = [
        r for r in project.regions
        if r.status in ("ocr_done", "detected", "suspected_text")
    ]

    def run_detect(progress_callback=None):
        result = pipeline.detect(project_id, detect_small_text=detect_small_text)
        return {"project_id": project_id, "regions": len(result.regions)}

    job = jm.submit("detect", project_id, run_detect)
    return job.to_dict()


@router.post("/{project_id}/jobs/correct")
async def submit_correct_job(project_id: str, auto_accept: bool = False):
    pipeline = get_pipeline()
    project = pipeline.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)

    jm = get_job_manager(pipeline)

    regions = [
        r for r in project.regions
        if r.status in ("ocr_done", "detected", "suspected_text")
    ]

    def run_correct(progress_callback=None):
        result = pipeline.correct(
            project_id,
            auto_accept=auto_accept,
            progress_callback=progress_callback,
        )
        llm_corrected = sum(1 for r in result.regions if r.status == "llm_corrected")
        return {"project_id": project_id, "corrected": llm_corrected}

    job = jm.submit("correct", project_id, run_correct)
    return job.to_dict()


@router.post("/{project_id}/jobs/inpaint")
async def submit_inpaint_job(project_id: str, method: str = "auto"):
    pipeline = get_pipeline()
    project = pipeline.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)

    jm = get_job_manager(pipeline)

    def run_inpaint(progress_callback=None):
        simple_mode = method == "simple"
        result = pipeline.inpaint(project_id, simple_mode=simple_mode)
        inpainted = sum(1 for r in result.regions if r.status == "inpainted")
        return {"project_id": project_id, "inpainted": inpainted}

    job = jm.submit("inpaint", project_id, run_inpaint)
    return job.to_dict()


@router.post("/{project_id}/jobs/render")
async def submit_render_job(project_id: str):
    pipeline = get_pipeline()
    project = pipeline.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)

    jm = get_job_manager(pipeline)

    def run_render(progress_callback=None):
        result = pipeline.render(project_id)
        rendered = sum(1 for r in result.regions if r.status == "rendered")
        return {"project_id": project_id, "rendered": rendered}

    job = jm.submit("render", project_id, run_render)
    return job.to_dict()
