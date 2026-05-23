from __future__ import annotations
import asyncio
import json
import logging
import queue
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.core.pipeline import Pipeline
from backend.security import require_api_token
from backend.exceptions import ProjectNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/projects",
    tags=["correction"],
    dependencies=[Depends(require_api_token)],
)


class CorrectRequest(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    auto_accept: bool = False


def get_pipeline() -> Pipeline:
    from backend.config import app_config
    return Pipeline(app_config)


@router.post("/{project_id}/correct")
async def correct_text(
    project_id: str,
    request: CorrectRequest = CorrectRequest(),
    pipeline: Pipeline = Depends(get_pipeline),
):
    """Run LLM correction with SSE progress streaming.

    Returns a streaming response with events:
      - data: {"type":"start","total":N}
      - data: {"type":"progress","completed":X,"total":N}
      - data: {"type":"complete","regions":[...]}
    """
    try:
        project = pipeline.project_store.load(project_id)
        if not project:
            raise ProjectNotFoundError(project_id)
    except ValueError:
        raise ProjectNotFoundError(project_id)

    total = len([r for r in project.regions
                 if r.status in ("ocr_done", "detected", "suspected_text")])

    progress_queue = queue.Queue()

    def progress_cb(completed, total_):
        progress_queue.put(completed)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = loop.run_in_executor(
                executor,
                lambda: pipeline.correct(
                    project_id,
                    auto_accept=request.auto_accept,
                    provider=request.provider,
                    model=request.model,
                    progress_callback=progress_cb,
                ),
            )

            last_completed = 0
            while not future.done():
                try:
                    completed = progress_queue.get(timeout=0.1)
                    last_completed = completed
                    yield f"data: {json.dumps({'type': 'progress', 'completed': completed, 'total': total})}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.05)

            try:
                result = future.result()
                yield f"data: {json.dumps({'type': 'complete', 'regions': [r.to_dict() for r in result.regions]})}\n\n"
            except Exception as e:
                logger.error(f"Correction failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
