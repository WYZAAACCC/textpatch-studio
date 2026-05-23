from __future__ import annotations
import logging

from backend.llm_adapters.base import LLMClient, TextCorrectionRequest, TextCorrectionResponse

logger = logging.getLogger(__name__)


class MockLLMClient(LLMClient):
    def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        return TextCorrectionResponse(
            corrected_text=request.ocr_best,
            confidence=0.0,
            correction_type="unavailable",
            changed_chars=[],
            uncertain_chars=[],
            needs_human=True,
            raw_response={"mock": True, "reason": "LLM API key not configured"},
        )
