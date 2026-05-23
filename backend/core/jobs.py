"""Unified job manager for long-running pipeline tasks.

Provides queued execution with progress tracking, cancellation,
and SSE event streaming for detect/ocr/correct/inpaint/render stages.
"""
from __future__ import annotations
import asyncio
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class Job:
    job_id: str
    type: str
    project_id: str
    status: JobStatus = JobStatus.queued
    progress_completed: int = 0
    progress_total: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    cancel_requested: bool = False
    _event_queue: queue.Queue = field(default_factory=queue.Queue)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "type": self.type,
            "project_id": self.project_id,
            "status": self.status.value,
            "progress": {
                "completed": self.progress_completed,
                "total": self.progress_total,
            },
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    def __init__(self, pipeline, max_concurrent: int = 2):
        self._pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._futures: dict[str, any] = {}

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, project_id: str = None) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
            if project_id:
                jobs = [j for j in jobs if j.project_id == project_id]
            return [j.to_dict() for j in jobs]

    def submit(self, job_type: str, project_id: str,
               fn: Callable, on_done: Callable = None) -> Job:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = Job(job_id=job_id, type=job_type, project_id=project_id)

        with self._lock:
            self._jobs[job_id] = job

        def _run():
            try:
                job.status = JobStatus.running
                job.started_at = _now()
                job._event_queue.put({"type": "start", "job_id": job_id})

                def progress_callback(completed: int, total: int):
                    job.progress_completed = completed
                    job.progress_total = total
                    job._event_queue.put({
                        "type": "progress",
                        "completed": completed,
                        "total": total,
                    })

                result = fn(progress_callback=progress_callback)
                job.result = result
                job.status = JobStatus.succeeded
                job._event_queue.put({"type": "complete", "result": result})
                if on_done:
                    on_done(job)
            except Exception as e:
                if job.cancel_requested:
                    job.status = JobStatus.cancelled
                    job._event_queue.put({"type": "cancelled"})
                else:
                    logger.error(f"Job {job_id} failed: {e}")
                    job.error = str(e)[:500]
                    job.status = JobStatus.failed
                    job._event_queue.put({"type": "error", "message": str(e)[:500]})
                if on_done:
                    on_done(job)
            finally:
                job.finished_at = _now()
                job._event_queue.put({
                    "type": "done",
                    "status": job.status.value,
                    "finished_at": job.finished_at,
                })

        future = self._executor.submit(_run)
        with self._lock:
            self._futures[job_id] = future
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status not in (JobStatus.queued, JobStatus.running):
                return False
            job.cancel_requested = True
        return True

    async def event_stream(self, job_id: str):
        job = self.get_job(job_id)
        if job is None:
            yield f"data: {{\"type\":\"error\",\"message\":\"Job not found\"}}\n\n"
            return

        import json as _json
        while job.status in (JobStatus.queued, JobStatus.running):
            try:
                event = job._event_queue.get(timeout=0.1)
                yield f"data: {_json.dumps(event, default=str)}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

        while not job._event_queue.empty():
            event = job._event_queue.get_nowait()
            yield f"data: {_json.dumps(event, default=str)}\n\n"

        yield f"data: {_json.dumps({'type': 'done', 'status': job.status.value, 'finished_at': job.finished_at})}\n\n"

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_job_manager: Optional[JobManager] = None


def get_job_manager(pipeline=None) -> JobManager:
    global _job_manager
    if _job_manager is None and pipeline is not None:
        _job_manager = JobManager(pipeline)
    return _job_manager
