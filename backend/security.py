from fastapi import Header, HTTPException, Depends

from backend.config import app_config


def require_api_token(
    authorization: str | None = Header(default=None),
    x_textpatch_token: str | None = Header(default=None, alias="X-TextPatch-Token"),
):
    if not app_config.security.require_auth:
        return

    expected = app_config.security.api_token
    if not expected:
        raise HTTPException(500, "Auth is required but TEXTPATCH_API_TOKEN is not set")

    token = x_textpatch_token
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]

    if token != expected:
        raise HTTPException(401, "Invalid or missing API token")
