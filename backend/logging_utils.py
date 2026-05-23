from __future__ import annotations
import logging
import os

MAX_OCR_LOG_CHARS = 80


def sanitize_log_message(msg: str) -> str:
    for var in ("DEEPSEEK_API_KEY", "TEXTPATCH_API_TOKEN", "TEXTPATCH_OCR_TOKEN"):
        val = os.environ.get(var, "")
        if val and len(val) > 4:
            msg = msg.replace(val, f"[REDACTED_{var}]")
    return msg


def truncate_ocr_text(text: str, max_chars: int = MAX_OCR_LOG_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[TRUNCATED]"


class SanitizingFormatter(logging.Formatter):
    def format(self, record):
        if record.msg and isinstance(record.msg, str):
            record = logging.makeLogRecord(vars(record))
            record.msg = sanitize_log_message(record.msg)
        return super().format(record)
