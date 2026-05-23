import pytest
import json
import tempfile
from pathlib import Path

from backend.models.project import Project
from backend.models.region import TextRegion
from backend.models.style import TextStyle
from backend.models.ocr import OCRInfo, OCRCandidate
from backend.models.llm import LLMCorrectionInfo, ChangedChar
from backend.models.render import RenderInfo
from backend.storage.project_store import ProjectStore
from backend.storage.file_store import FileStore


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def project_store(tmp_dir):
    return ProjectStore(tmp_dir / "projects")


@pytest.fixture
def file_store(tmp_dir):
    return FileStore(tmp_dir / "projects")


@pytest.fixture
def sample_project():
    project = Project.create(
        name="test_project",
        image_path="/tmp/test.png",
        width=800,
        height=600,
    )
    region = TextRegion.create(
        polygon=[[10, 10], [200, 10], [200, 50], [10, 50]],
        bbox=[10, 10, 200, 50],
        confidence=0.9,
    )
    region.final_text = "测试文字"
    region.ocr = OCRInfo(
        best_text="测试文字",
        confidence=0.9,
        candidates=[OCRCandidate(text="测试文字", confidence=0.9, source="1x")],
    )
    region.llm = LLMCorrectionInfo(
        provider="deepseek",
        model="deepseek-v4-flash",
        suggested_text="测试文字",
        confidence=0.95,
        correction_type="unchanged",
        changed_chars=[],
        needs_human=False,
    )
    region.style = TextStyle(font_size=24)
    region.render = RenderInfo()
    project.regions = [region]
    return project


class TestProjectStore:
    def test_save_and_load(self, project_store, sample_project):
        project_store.save(sample_project)
        loaded = project_store.load(sample_project.id)
        assert loaded is not None
        assert loaded.id == sample_project.id
        assert loaded.name == "test_project"
        assert loaded.width == 800
        assert loaded.height == 600
        assert len(loaded.regions) == 1

    def test_load_nonexistent(self, project_store):
        loaded = project_store.load("nonexistent")
        assert loaded is None

    def test_delete(self, project_store, sample_project):
        project_store.save(sample_project)
        assert project_store.delete(sample_project.id) is True
        assert project_store.load(sample_project.id) is None

    def test_list_projects(self, project_store, sample_project):
        project_store.save(sample_project)
        projects = project_store.list_projects()
        assert len(projects) == 1
        assert projects[0]["id"] == sample_project.id

    def test_project_exists(self, project_store, sample_project):
        project_store.save(sample_project)
        assert project_store.project_exists(sample_project.id) is True
        assert project_store.project_exists("nonexistent") is False


class TestProjectSerialization:
    def test_to_dict(self, sample_project):
        d = sample_project.to_dict()
        assert d["id"] == sample_project.id
        assert d["name"] == "test_project"
        assert len(d["regions"]) == 1
        assert d["regions"][0]["final_text"] == "测试文字"

    def test_from_dict(self, sample_project):
        d = sample_project.to_dict()
        restored = Project.from_dict(d)
        assert restored.id == sample_project.id
        assert restored.name == "test_project"
        assert len(restored.regions) == 1
        assert restored.regions[0].final_text == "测试文字"

    def test_round_trip(self, sample_project):
        d = sample_project.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        parsed = json.loads(json_str)
        restored = Project.from_dict(parsed)
        assert restored.id == sample_project.id
        assert len(restored.regions) == len(sample_project.regions)


class TestRegionSerialization:
    def test_region_to_dict(self):
        region = TextRegion.create(
            polygon=[[10, 10], [200, 10], [200, 50], [10, 50]],
            bbox=[10, 10, 200, 50],
            confidence=0.9,
            is_tiny=True,
        )
        region.final_text = "测试"
        region.risk_flags = ["number"]

        d = region.to_dict()
        assert d["id"].startswith("region_")
        assert d["is_tiny"] is True
        assert d["final_text"] == "测试"
        assert "number" in d["risk_flags"]

    def test_region_from_dict(self):
        data = {
            "id": "region_test",
            "polygon": [[10, 10], [200, 10], [200, 50], [10, 50]],
            "bbox": [10, 10, 200, 50],
            "angle": 0.0,
            "source": "detection",
            "confidence": 0.9,
            "is_tiny": True,
            "status": "ocr_done",
            "ocr": {
                "best_text": "测试",
                "confidence": 0.9,
                "candidates": [{"text": "测试", "confidence": 0.9, "source": "1x"}],
            },
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "suggested_text": "测试",
                "confidence": 0.95,
                "correction_type": "unchanged",
                "changed_chars": [],
                "needs_human": False,
                "raw_response": {},
            },
            "final_text": "测试",
            "risk_flags": ["number"],
            "review": {"status": "pending", "reviewer": "", "comment": "", "reviewed_at": ""},
            "style": {
                "font_family": "Noto Sans CJK SC",
                "font_size": 24.0,
                "color": [0, 0, 0, 255],
            },
            "render": {"transform": "none", "blend_mode": "normal", "overflow": False},
        }
        region = TextRegion.from_dict(data)
        assert region.id == "region_test"
        assert region.is_tiny is True
        assert region.ocr.best_text == "测试"
        assert region.llm.provider == "deepseek"
        assert region.style.font_size == 24.0
        assert "number" in region.risk_flags


class TestFileStore:
    def test_is_allowed_file(self):
        assert FileStore.is_allowed_file("test.jpg") is True
        assert FileStore.is_allowed_file("test.png") is True
        assert FileStore.is_allowed_file("test.webp") is True
        assert FileStore.is_allowed_file("test.tiff") is True
        assert FileStore.is_allowed_file("test.gif") is False
        assert FileStore.is_allowed_file("test.pdf") is False

    def test_save_and_get_original(self, file_store, tmp_dir):
        project_id = "test_project_001"
        img_path = tmp_dir / "test_input.png"
        img = __import__("PIL").Image.new("RGB", (800, 600), (255, 255, 255))
        img.save(str(img_path))

        original = file_store.save_original(project_id, img_path, "test.png")
        assert original.exists()

        got = file_store.get_original(project_id)
        assert got is not None
        assert got.exists()
