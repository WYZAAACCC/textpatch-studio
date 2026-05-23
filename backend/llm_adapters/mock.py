from __future__ import annotations
import logging

from backend.llm_adapters.base import LLMClient, TextCorrectionRequest, TextCorrectionResponse

logger = logging.getLogger(__name__)


class MockLLMClient(LLMClient):
    def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        best = request.ocr_best
        needs_human = False

        if request.risk_flags:
            needs_human = True

        return TextCorrectionResponse(
            corrected_text=best,
            confidence=0.95,
            correction_type="unchanged",
            changed_chars=[],
            uncertain_chars=[],
            needs_human=needs_human,
            raw_response={"mock": True},
        )
