"""Tests for path safety: project_id/region_id validation, path traversal protection."""
import pytest
from pathlib import Path

from backend.storage.path_safety import (
    validate_project_id,
    validate_region_id,
    ensure_child_path,
    PROJECT_ID_RE,
    REGION_ID_RE,
)


class TestProjectIdValidation:
    def test_valid_project_id(self):
        validate_project_id("p_20260425_37c77c01")

    def test_invalid_project_id_format(self):
        with pytest.raises(ValueError):
            validate_project_id("invalid")

    def test_empty_project_id(self):
        with pytest.raises(ValueError):
            validate_project_id("")

    def test_path_traversal_in_project_id(self):
        with pytest.raises(ValueError):
            validate_project_id("../../../etc/passwd")

    def test_project_id_with_spaces(self):
        with pytest.raises(ValueError):
            validate_project_id("p_20260425_37c77c01 ")


class TestRegionIdValidation:
    def test_valid_region_id(self):
        validate_region_id("region_a0fe77")

    def test_invalid_region_id(self):
        with pytest.raises(ValueError):
            validate_region_id("not_a_region")

    def test_path_traversal_in_region_id(self):
        with pytest.raises(ValueError):
            validate_region_id("region_../etc")


class TestEnsureChildPath:
    def test_valid_child(self, tmp_path):
        base = tmp_path / "data"
        base.mkdir()
        child = base / "projects"
        result = ensure_child_path(base, child)
        assert result == child

    def test_path_traversal_blocked(self, tmp_path):
        base = tmp_path / "data"
        base.mkdir()
        sibling = tmp_path / "evil"
        sibling.mkdir()
        with pytest.raises(ValueError, match="Unsafe"):
            ensure_child_path(base, sibling)

    def test_exact_base_ok(self, tmp_path):
        base = tmp_path / "data"
        base.mkdir()
        result = ensure_child_path(base, base)
        assert result == base
