from fastapi import Header, HTTPException

from backend.config import app_config


def require_api_token(
    authorization: str | None = Header(default=None),
    x_textpatch_token: str | None = Header(default=None, alias="X-TextPatch-Token"),
):
    if not app_config.security.require_auth:
        return

    expected = app_config.security.api_token
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Auth is required but TEXTPATCH_API_TOKEN is not set",
        )

    token = x_textpatch_token
    if authorization and str(authorization).lower().startswith("bearer "):
        token = str(authorization)[7:]

    if token != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token",
        )
