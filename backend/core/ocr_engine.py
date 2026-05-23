from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np

from backend.models.region import TextRegion
from backend.models.ocr import OCRInfo, OCRCandidate

logger = logging.getLogger(__name__)


def run_ocr(
    image: np.ndarray,
    regions: list[TextRegion],
    ocr_engine=None,
) -> list[TextRegion]:
    if ocr_engine is None:
        logger.warning("No OCR engine provided, skipping OCR")
        return regions

    for region in regions:
        try:
            if region.final_text and region.confidence > 0.5:
                region.ocr = OCRInfo(
                    best_text=region.final_text,
                    confidence=region.confidence,
                    candidates=[
                        OCRCandidate(
                            text=region.final_text,
                            confidence=region.confidence,
                            source=region.source,
                        )
                    ],
                )
                if region.status in ("detected", "suspected_text"):
                    region.status = "ocr_done"
                continue

            ocr_info = _ocr_region(image, region, ocr_engine)
            region.ocr = ocr_info
            if ocr_info.best_text:
                region.final_text = ocr_info.best_text
                if region.status in ("detected", "suspected_text"):
                    region.status = "ocr_done"
            else:
                if region.is_tiny:
                    region.status = "needs_manual_input"
                else:
                    region.status = "ocr_done"
        except Exception as e:
            logger.error(f"OCR failed for region {region.id}: {e}")
            region.status = "needs_manual_input"

    for i, region in enumerate(regions):
        if region.final_text:
            regions[i] = _postprocess_region(region)

    return regions


def _ocr_region(
    image: np.ndarray, region: TextRegion, ocr_engine
) -> OCRInfo:
    candidates = []

    if not region.bbox or len(region.bbox) < 4:
        return OCRInfo(best_text="", confidence=0.0, candidates=candidates)

    x1, y1, x2, y2 = [int(v) for v in region.bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return OCRInfo(best_text="", confidence=0.0, candidates=candidates)

    crop = image[y1:y2, x1:x2]

    if region.final_text:
        candidates.append(
            OCRCandidate(
                text=region.final_text,
                confidence=region.confidence,
                source="detection",
            )
        )

    try:
        _, img_encoded = cv2.imencode(".png", crop)
        image_bytes = img_encoded.tobytes()

        results = ocr_engine.recognize(image_bytes, file_type=1)
        for res in results:
            if res.text.strip():
                candidates.append(
                    OCRCandidate(
                        text=res.text.strip(),
                        confidence=res.confidence,
                        source="ocr_recheck",
                    )
                )
    except Exception as e:
        logger.warning(f"OCR recheck failed for region {region.id}: {e}")

    if not candidates:
        return OCRInfo(best_text="", confidence=0.0, candidates=candidates)

    best = max(candidates, key=lambda c: c.confidence)
    merged_candidates = _merge_candidates(candidates)

    from backend.core.ocr_postprocess import merge_ocr_candidates, clean_ocr_artifacts

    best_text = best.text
    best_text = clean_ocr_artifacts(best_text)

    if len(merged_candidates) > 1:
        candidate_texts = [c.text for c in merged_candidates[:5]]
        candidate_confs = [c.confidence for c in merged_candidates[:5]]
        merged_text = merge_ocr_candidates(candidate_texts, candidate_confs)
        if len(merged_text) >= len(best_text) * 0.8:
            best_text = merged_text

    return OCRInfo(
        best_text=best_text,
        confidence=best.confidence,
        candidates=merged_candidates,
    )


def _merge_candidates(candidates: list[OCRCandidate]) -> list[OCRCandidate]:
    text_groups: dict[str, list[OCRCandidate]] = {}
    for c in candidates:
        key = c.text
        if key in text_groups:
            text_groups[key].append(c)
        else:
            text_groups[key] = [c]

    merged = []
    for text, group in text_groups.items():
        avg_conf = sum(c.confidence for c in group) / len(group)
        sources = ", ".join(set(c.source for c in group))
        merged.append(
            OCRCandidate(text=text, confidence=avg_conf, source=sources)
        )

    merged.sort(key=lambda c: c.confidence, reverse=True)
    return merged


def _postprocess_region(region: TextRegion) -> TextRegion:
    from backend.core.ocr_postprocess import validate_ocr_text, detect_pseudo_hanzi_in_text

    validation = validate_ocr_text(region.final_text)

    if not validation["valid"]:
        if "pseudo_hanzi" in validation["issues"]:
            pseudo_chars = validation.get("pseudo_chars", [])
            if pseudo_chars:
                if not region.risk_flags:
                    region.risk_flags = []
                region.risk_flags.append("pseudo_hanzi")

                if validation["cleaned"] and len(validation["cleaned"]) > 0:
                    region.final_text = validation["cleaned"]

        if "ocr_artifacts" in validation["issues"]:
            if validation["cleaned"]:
                region.final_text = validation["cleaned"]

    return region
