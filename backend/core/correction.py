from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from backend.models.region import TextRegion
from backend.models.llm import LLMCorrectionInfo, ChangedChar
from backend.llm_adapters.base import LLMClient, TextCorrectionRequest

logger = logging.getLogger(__name__)


def correct_regions(
    regions: list[TextRegion],
    llm_client: LLMClient,
    auto_accept: bool = False,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[TextRegion]:
    # Collect regions eligible for correction
    region_indices = [
        (i, r) for i, r in enumerate(regions)
        if r.status in ("ocr_done", "detected", "suspected_text")
    ]

    total = len(region_indices)
    if total == 0:
        return regions

    results = list(regions)  # shallow copy for safe mutation
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_correct_region, r, llm_client, auto_accept, regions): i
            for i, r in region_indices
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"LLM correction failed for region {regions[i].id}: {e}")
                regions[i].status = "needs_review"
                regions[i].llm = LLMCorrectionInfo(
                    provider="error",
                    model="",
                    suggested_text=regions[i].final_text,
                    confidence=0.0,
                    correction_type="uncertain",
                    changed_chars=[],
                    needs_human=True,
                    raw_response={"error": str(e)},
                )
                results[i] = regions[i]
            completed += 1
            if progress_callback:
                try:
                    progress_callback(completed, total)
                except Exception:
                    pass  # don't let callback failures break the pipeline

    return results


def _correct_region(
    region: TextRegion, llm_client: LLMClient, auto_accept: bool,
    all_regions: list[TextRegion] = None,
) -> TextRegion:
    ocr_best = region.final_text
    if not ocr_best and region.ocr:
        ocr_best = region.ocr.best_text

    if not ocr_best:
        region.status = "needs_manual_input"
        return region

    ocr_candidates = []
    if region.ocr and region.ocr.candidates:
        ocr_candidates = region.ocr.candidates

    neighbor_texts = _get_neighbor_texts(region, all_regions)

    request = TextCorrectionRequest(
        ocr_best=ocr_best,
        ocr_candidates=ocr_candidates,
        neighbor_texts=neighbor_texts,
        risk_flags=region.risk_flags,
    )

    response = llm_client.correct_text(request)

    changed_chars = []
    for cc in response.changed_chars:
        if isinstance(cc, dict):
            changed_chars.append(
                ChangedChar(
                    from_char=cc.get("from_char", ""),
                    to_char=cc.get("to_char", ""),
                    reason=cc.get("reason", ""),
                )
            )
        elif isinstance(cc, ChangedChar):
            changed_chars.append(cc)

    llm_info = LLMCorrectionInfo(
        provider=getattr(llm_client, "model", "unknown"),
        model=getattr(llm_client, "model", "unknown"),
        suggested_text=response.corrected_text,
        confidence=response.confidence,
        correction_type=response.correction_type,
        changed_chars=changed_chars,
        needs_human=response.needs_human,
        raw_response=response.raw_response,
    )

    region.llm = llm_info

    # ── Apply LLM formula detection ──
    if response.is_formula and response.latex:
        region.is_formula = True
        region.latex_source = response.latex
        # Use LaTeX as the final text for formula regions
        region.final_text = response.latex
    else:
        # Fall back to heuristic formula classification
        from backend.renderer.formula_renderer import classify_as_formula
        is_formula, reason = classify_as_formula(response.corrected_text)
        if is_formula:
            region.is_formula = True
            region.latex_source = response.corrected_text
            region.final_text = response.corrected_text

    if not response.is_formula:
        region.final_text = response.corrected_text

    from backend.core.risk_rules import should_auto_accept, should_force_review, is_forbidden_change

    if is_forbidden_change(ocr_best, response.corrected_text):
        region.status = "needs_review"
        region.risk_flags.append("forbidden_change")
        return region

    if auto_accept and should_auto_accept(
        ocr_best, response.corrected_text, region.ocr.confidence if region.ocr else 0,
        response.confidence, response.needs_human, region.bbox
    ):
        region.status = "approved"
    elif should_force_review(
        ocr_best, response.corrected_text, region.ocr.confidence if region.ocr else 0,
        response.confidence, response.needs_human, region.bbox
    ):
        region.status = "needs_review"
    else:
        region.status = "llm_corrected"

    return region


def _get_neighbor_texts(region: TextRegion, all_regions: list[TextRegion] = None, max_neighbors: int = 4) -> list[str]:
    if not all_regions or not region.bbox or len(region.bbox) < 4:
        return []

    x1, y1, x2, y2 = region.bbox
    region_cx = (x1 + x2) / 2
    region_cy = (y1 + y2) / 2

    neighbors = []
    for other in all_regions:
        if other is region or not other.final_text or not other.bbox or len(other.bbox) < 4:
            continue

        ox1, oy1, ox2, oy2 = other.bbox
        other_cx = (ox1 + ox2) / 2
        other_cy = (oy1 + oy2) / 2

        dx = region_cx - other_cx
        dy = region_cy - other_cy
        dist = (dx * dx + dy * dy) ** 0.5

        max_dim = max(x2 - x1, y2 - y1)
        if dist < max_dim * 10:
            neighbors.append((dist, other.final_text))

    neighbors.sort(key=lambda x: x[0])
    return [text for _, text in neighbors[:max_neighbors]]
