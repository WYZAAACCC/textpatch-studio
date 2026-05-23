from __future__ import annotations
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from datetime import datetime

from backend.models.project import Project
from backend.storage.path_safety import (
    validate_project_id,
    validate_region_id,
    ensure_child_path,
)


class ProjectStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_lock = threading.Lock()

    def _get_lock(self, project_id: str) -> threading.RLock:
        with self._locks_lock:
            if project_id not in self._locks:
                self._locks[project_id] = threading.RLock()
            return self._locks[project_id]

    @contextmanager
    def lock_project(self, project_id: str):
        lock = self._get_lock(project_id)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _project_dir(self, project_id: str, create: bool = False) -> Path:
        validate_project_id(project_id)
        p = self.data_dir / f"{project_id}.textpatch"
        p = ensure_child_path(self.data_dir, p)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def _project_json(self, project_id: str, create: bool = False) -> Path:
        return self._project_dir(project_id, create=create) / "project.json"

    def _regions_dir(self, project_id: str, create: bool = False) -> Path:
        d = self._project_dir(project_id, create=create) / "regions"
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, project: Project) -> None:
        validate_project_id(project.id)
        with self.lock_project(project.id):
            project.updated_at = datetime.now().isoformat()
            json_path = self._project_json(project.id, create=True)
            data = project.to_dict()
            tmp = json_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, json_path)

    def load(self, project_id: str) -> Optional[Project]:
        json_path = self._project_json(project_id, create=False)
        if not json_path.exists():
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Project.from_dict(data)

    def delete(self, project_id: str) -> bool:
        validate_project_id(project_id)
        with self.lock_project(project_id):
            project_dir = self._project_dir(project_id)
            if not project_dir.exists():
                return False
            import shutil
            shutil.rmtree(project_dir)
            with self._locks_lock:
                self._locks.pop(project_id, None)
            return True

    def list_projects(self) -> list[dict]:
        results = []
        for d in self.data_dir.iterdir():
            if d.is_dir() and d.suffix == ".textpatch":
                json_path = d / "project.json"
                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    results.append({
                        "id": data["id"],
                        "name": data["name"],
                        "width": data["width"],
                        "height": data["height"],
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "region_count": len(data.get("regions", [])),
                    })
        return results

    def get_region_path(self, project_id: str, region_id: str, suffix: str) -> Path:
        validate_project_id(project_id)
        validate_region_id(region_id)
        regions_dir = self._regions_dir(project_id, create=False)
        path = ensure_child_path(regions_dir, regions_dir / f"{region_id}_{suffix}")
        return path

    def project_exists(self, project_id: str) -> bool:
        return self._project_json(project_id).exists()
