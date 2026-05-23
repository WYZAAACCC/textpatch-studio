"""Tests for security module: auth, CORS, host binding."""
import os
import pytest
from unittest.mock import patch


class TestSecurityConfig:
    def test_default_host_is_localhost(self):
        from backend.config import AppConfig
        config = AppConfig()
        assert config.host == "127.0.0.1"

    def test_default_auth_disabled(self):
        from backend.config import SecurityConfig
        sec = SecurityConfig()
        assert sec.require_auth is False
        assert sec.api_token == ""

    def test_default_cors_localhost(self):
        from backend.config import SecurityConfig
        sec = SecurityConfig()
        assert "http://127.0.0.1:8000" in sec.allowed_origins
        assert "http://localhost:8000" in sec.allowed_origins

    def test_storage_defaults(self):
        from backend.config import StorageConfig
        sc = StorageConfig()
        assert sc.max_file_size_mb == 50
        assert sc.max_image_pixels == 40_000_000
        assert sc.max_image_width == 12000
        assert sc.max_image_height == 12000


class TestRequireApiToken:
    def test_no_auth_required_passes(self):
        with patch("backend.security.app_config") as mock_cfg:
            mock_cfg.security.require_auth = False
            from backend.security import require_api_token
            result = require_api_token(authorization=None, x_textpatch_token=None)
            assert result is None

    def test_auth_required_no_token_raises(self):
        from fastapi import HTTPException
        with patch("backend.security.app_config") as mock_cfg:
            mock_cfg.security.require_auth = True
            mock_cfg.security.api_token = "secret"
            from backend.security import require_api_token
            try:
                require_api_token(authorization=None, x_textpatch_token=None)
                assert False, "Should have raised"
            except HTTPException as e:
                assert e.status_code == 401

    def test_auth_required_correct_token_passes(self):
        with patch("backend.security.app_config") as mock_cfg:
            mock_cfg.security.require_auth = True
            mock_cfg.security.api_token = "secret"
            from backend.security import require_api_token
            result = require_api_token(
                authorization=None, x_textpatch_token="secret"
            )
            assert result is None

    def test_bearer_token_accepted(self):
        with patch("backend.security.app_config") as mock_cfg:
            mock_cfg.security.require_auth = True
            mock_cfg.security.api_token = "secret"
            from backend.security import require_api_token
            result = require_api_token(
                authorization="Bearer secret", x_textpatch_token=None
            )
            assert result is None

    def test_wrong_token_raises(self):
        from fastapi import HTTPException
        with patch("backend.security.app_config") as mock_cfg:
            mock_cfg.security.require_auth = True
            mock_cfg.security.api_token = "secret"
            from backend.security import require_api_token
            try:
                require_api_token(x_textpatch_token="wrong")
                assert False, "Should have raised"
            except HTTPException as e:
                assert e.status_code == 401

    def test_no_token_configured_raises_500(self):
        from fastapi import HTTPException
        with patch("backend.security.app_config") as mock_cfg:
            mock_cfg.security.require_auth = True
            mock_cfg.security.api_token = ""
            from backend.security import require_api_token
            try:
                require_api_token(authorization=None, x_textpatch_token=None)
                assert False, "Should have raised"
            except HTTPException as e:
                assert e.status_code == 500
