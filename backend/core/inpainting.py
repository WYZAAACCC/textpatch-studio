from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend.models.region import TextRegion
from backend.core.masking import generate_masks
from backend.core.style_inference import infer_style
from backend.core.border_detection import is_border_region
from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
from backend.inpaint_adapters.simple_fill import SimpleFillEngine

logger = logging.getLogger(__name__)


def inpaint_regions(
    image: np.ndarray,
    regions: list[TextRegion],
    inpaint_engine: Optional[OpenCVInpaintEngine] = None,
    simple_mode: bool = False,
) -> tuple[np.ndarray, list[TextRegion]]:
    if inpaint_engine is None:
        inpaint_engine = OpenCVInpaintEngine()

    if simple_mode:
        # Tight fill: smaller dilation, no GrabCut — fill only glyph + shadow
        simple_engine = SimpleFillEngine(shadow_dilation=1)
    else:
        simple_engine = SimpleFillEngine(shadow_dilation=2)
    result = image.copy()

    border_skipped = 0
    solid_filled = 0
    tealea_used = 0

    for region in regions:
        if region.status not in ("approved", "needs_review", "llm_corrected"):
            continue

        try:
            if region.style is None:
                region.style = infer_style(image, region)

            glyph_mask, erase_mask, safe_mask = generate_masks(
                image, region, use_grabcut=not simple_mode
            )

            # ---- Region-level border detection ----
            # Skip entire regions that are structural elements (lines,
            # boxes, borders) rather than text. This is a coarse filter:
            # either the whole region is structural, or it's all text.
            # We do NOT do pixel-level protection which causes partial
            # erasure of text characters near structural lines.
            if is_border_region(region, image):
                border_skipped += 1
                region.status = "inpainted"
                continue

            if not np.any(erase_mask > 0):
                region.status = "inpainted"
                continue

            # ---- Erasure ----
            if simple_mode:
                result = simple_engine.inpaint(
                    result, erase_mask, region_bbox=region.bbox
                )
                solid_filled += 1
            elif _needs_telea(image, erase_mask, region):
                result = _replace_text_with_opencv_inpaint(
                    result, erase_mask, inpaint_engine
                )
                tealea_used += 1
            else:
                result = simple_engine.inpaint(
                    result, erase_mask, region_bbox=region.bbox
                )
                solid_filled += 1

            region.status = "inpainted"

        except Exception as e:
            logger.error(f"Inpaint failed for region {region.id}: {e}")

    if border_skipped or solid_filled or tealea_used:
        logger.info(
            f"Inpaint: {border_skipped} borders skipped, "
            f"{solid_filled} solid-fill, {tealea_used} TELEA"
        )

    return result, regions


def _replace_text_with_opencv_inpaint(
    image: np.ndarray,
    erase_mask: np.ndarray,
    inpaint_engine,
) -> np.ndarray:
    return inpaint_engine.inpaint(image, erase_mask)


def _needs_telea(
    image: np.ndarray,
    erase_mask: np.ndarray,
    region: TextRegion,
) -> bool:
    """Check if the background is complex enough to need OpenCV TELEA.

    Only returns True for truly complex / photographic backgrounds with
    very high colour variance AND high edge density. Scientific flowchart
    diagrams almost never trigger this.
    """
    h, w = image.shape[:2]

    if not region.bbox or len(region.bbox) < 4:
        return False

    x1, y1, x2, y2 = [int(v) for v in region.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(erase_mask, kernel).astype(bool)

    pad = max(4, int(max(x2 - x1, y2 - y1) * 0.15))
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(w, x2 + pad), min(h, y2 + pad)

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[by1:by2, bx1:bx2] = True
    border_mask = border_mask & ~dilated

    bg_pixel_count = np.sum(border_mask)
    if bg_pixel_count < 50:
        return False

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    bg_vals = gray[border_mask].astype(np.float64)
    variance = float(np.var(bg_vals))

    # Only trigger TELEA for truly complex/photographic backgrounds.
    # Flowchart blocks with solid colours have variance < 2000 and
    # edge density < 0.20. Using TELEA on them produces white patches.
    if variance < 4000:
        return False

    crop = gray[by1:by2, bx1:bx2]
    edges = cv2.Canny(crop, 50, 150)
    crop_dilated = dilated[by1:by2, bx1:bx2]
    edges_clean = edges[~crop_dilated]
    if edges_clean.size == 0:
        return False
    edge_density = np.sum(edges_clean > 0) / edges_clean.size

    return edge_density > 0.30
