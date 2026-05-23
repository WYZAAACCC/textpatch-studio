from __future__ import annotations
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.core.jobs import get_job_manager, JobStatus
from backend.dependencies import get_pipeline
from backend.security import require_api_token

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/jobs",
    tags=["jobs"],
    dependencies=[Depends(require_api_token)],
)


@router.get("")
async def list_jobs(project_id: str = None):
    jm = get_job_manager()
    return {"jobs": jm.list_jobs(project_id=project_id)}


@router.get("/{job_id}")
async def get_job(job_id: str):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/{job_id}/events")
async def job_events(job_id: str):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        async for data in jm.event_stream(job_id):
            yield data

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    jm = get_job_manager()
    if not jm.cancel(job_id):
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    return {"status": "cancelling"}
