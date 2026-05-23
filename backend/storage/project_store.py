from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from backend.models.project import Project


class ProjectStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        p = self.data_dir / f"{project_id}.textpatch"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _project_json(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _regions_dir(self, project_id: str) -> Path:
        d = self._project_dir(project_id) / "regions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, project: Project) -> None:
        project.updated_at = datetime.now().isoformat()
        json_path = self._project_json(project.id)
        data = project.to_dict()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, project_id: str) -> Optional[Project]:
        json_path = self._project_json(project_id)
        if not json_path.exists():
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Project.from_dict(data)

    def delete(self, project_id: str) -> bool:
        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            return False
        import shutil
        shutil.rmtree(project_dir)
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
        return self._regions_dir(project_id) / f"{region_id}_{suffix}"

    def project_exists(self, project_id: str) -> bool:
        return self._project_json(project_id).exists()
