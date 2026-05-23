from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np

from backend.inpaint_adapters.base import InpaintEngine

logger = logging.getLogger(__name__)


class SimpleFillEngine(InpaintEngine):
    """Fill text regions with a uniform background colour.

    Samples a ring of pixels around the text mask, clusters them to find
    the dominant background colour, then fills the entire mask uniformly
    with that single colour.

    This produces perfectly uniform fills — no gradient, no blending
    artefacts, no colour variation within the erased area.
    """

    def __init__(self, shadow_dilation: int = 2, feather_width: int = 1):
        self.shadow_dilation = shadow_dilation
        self.feather_width = feather_width

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        region_bbox: Optional[tuple] = None,
        **kwargs,
    ) -> np.ndarray:
        if mask.dtype == np.uint8:
            mask_bool = mask > 0
        else:
            mask_bool = mask.astype(bool)

        if not np.any(mask_bool):
            return image

        result = image.copy()

        # ---- Step 1: Dilate mask to cover the shadow/glow ring ----
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.shadow_dilation * 2 + 1, self.shadow_dilation * 2 + 1),
        )
        fill_mask = cv2.dilate(
            mask.astype(np.uint8) * 255, dilate_kernel
        ).astype(bool)

        if not np.any(fill_mask):
            return result

        # ---- Step 2: Sample background with clustering ----
        fill_color = _sample_background_clustered(
            image, fill_mask, region_bbox
        )

        # ---- Step 2b: Validate against immediate boundary ----
        # If the sampled fill colour is suspiciously bright compared to
        # the pixels right next to the text, use the inner ring instead.
        fill_color = _validate_fill_color(image, fill_mask, fill_color)

        # ---- Step 3: Fill uniformly with single colour ----
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c][fill_mask] = fill_color[c]
        else:
            result[fill_mask] = fill_color

        # ---- Step 4: Feather edges (1px) to avoid hard boundary ----
        if self.feather_width > 0:
            result = _feather_mask_edge(result, fill_mask, self.feather_width)

        return result


def _validate_fill_color(
    image: np.ndarray,
    fill_mask: np.ndarray,
    fill_color: list,
) -> list:
    """Check that fill_color is consistent with the immediate text boundary.

    If the fill colour is near-white but the ring closest to the text is
    much darker, the sampling was contaminated by white background. Use
    the inner ring's median instead.
    """
    is_color = len(image.shape) == 3

    # Only validate if fill looks white-ish (low channel spread, high mean)
    if is_color:
        fill_span = max(fill_color) - min(fill_color)
        fill_mean = sum(fill_color) / 3.0
        if fill_span > 30 or fill_mean < 200:
            return fill_color
    else:
        if fill_color < 200:
            return fill_color

    # Sample the tightest ring (1-2px from fill_mask)
    inner = _sample_ring_pixels(image, fill_mask, ring_width=2)
    if inner is None or len(inner) < 5:
        return fill_color

    if is_color:
        inner_median = [int(np.median(inner[:, c])) for c in range(3)]
        inner_mean = sum(inner_median) / 3.0
    else:
        inner_median = int(np.median(inner))
        inner_mean = inner_median

    # If inner ring is significantly darker, use it
    if inner_mean < fill_mean - 20:
        return inner_median

    return fill_color


def _sample_background_clustered(
    image: np.ndarray,
    fill_mask: np.ndarray,
    region_bbox: Optional[tuple] = None,
) -> list:
    """Sample background colour by analysing pixels around the mask.

    Uses a tiered approach prioritising the innermost ring (closest
    to text), which is the most reliable indicator of the true
    background colour. Wider rings and context are only used as
    fallbacks when the inner ring has too few samples.
    """
    is_color = len(image.shape) == 3

    # Sample from multiple ring distances. AI-generated flowcharts often
    # have a white "halo" around text that extends 3-5 px past the
    # character edge. We start at 5 px from fill_mask (7 px from text)
    # to get past this halo into the true background.
    for ring_w in [5, 8, 12]:
        samples = _sample_ring_pixels(image, fill_mask, ring_width=ring_w)
        if samples is not None and len(samples) >= 10:
            if is_color:
                samples = samples.astype(np.float32)
            else:
                samples = samples.reshape(-1, 1).astype(np.float32)
            samples = _filter_outliers(samples, is_color)
            if len(samples) >= 10:
                color = _histogram_dominant_color(samples, is_color)
                if color is not None:
                    return color

    # Tier 4: Region context as last resort.
    ctx = _sample_context_pixels(image, fill_mask, region_bbox, padding_frac=0.12)
    if ctx is not None and len(ctx) >= 10:
        if is_color:
            ctx = ctx.astype(np.float32)
        else:
            ctx = ctx.reshape(-1, 1).astype(np.float32)
        ctx = _filter_outliers(ctx, is_color)
        if len(ctx) >= 10:
            color = _histogram_dominant_color(ctx, is_color)
            if color is not None:
                return color

    # Desperate fallback
    return _desperate_fallback(image, region_bbox, is_color)


def _filter_outliers(pixels: np.ndarray, is_color: bool) -> np.ndarray:
    """Remove outlier pixels that deviate significantly from the median."""
    if is_color:
        median = np.median(pixels, axis=0)
        diff = np.abs(pixels - median)
        std = np.std(pixels, axis=0)
        threshold = np.maximum(std * 1.5, 15.0)
        keep = np.all(diff < threshold, axis=1)
        return pixels[keep]
    else:
        median = np.median(pixels)
        diff = np.abs(pixels - median)
        std = float(np.std(pixels))
        threshold = max(std * 1.5, 15.0)
        return pixels[diff.flatten() < threshold]


def _histogram_dominant_color(
    pixels: np.ndarray,
    is_color: bool,
) -> Optional[list]:
    """Find the most frequent non-white colour using histogram quantization.

    Quantizes colours into bins, then picks the largest bin whose
    mean is not near-white. More robust than k-means when white
    background pixels dominate the sample.
    """
    if len(pixels) < 10:
        return None

    # Quantize to 32 levels per channel (8 levels of 32 each)
    if is_color:
        quantized = (pixels // 32).astype(np.int64)
        # Pack RGB into single 64-bit int for unique counting
        packed = (
            quantized[:, 0] * 65536
            + quantized[:, 1] * 256
            + quantized[:, 2]
        )
    else:
        quantized = (pixels // 32)
        packed = quantized.flatten().astype(np.int64)

    unique, inverse, counts = np.unique(
        packed, return_inverse=True, return_counts=True
    )

    # Sort bins by count (largest first)
    sorted_idx = np.argsort(counts)[::-1]

    for idx in sorted_idx:
        if counts[idx] < 5:
            continue

        bin_mask = inverse == idx
        bin_pixels = pixels[bin_mask]

        if is_color:
            bin_mean = float(np.mean(bin_pixels))
        else:
            bin_mean = float(np.mean(bin_pixels))

        # Skip near-white bins — background, not block fill.
        # White has low channel spread AND high per-channel values.
        if is_color:
            c_min = float(np.min(bin_pixels))
            c_max = float(np.max(bin_pixels))
            c_mean = float(np.mean(bin_pixels))
            if c_max - c_min < 30 and c_mean > 210:
                continue
        else:
            if bin_mean > 200:
                continue

        if is_color:
            return [int(np.median(bin_pixels[:, c])) for c in range(3)]
        else:
            return int(np.median(bin_pixels))

    # All bins are near-white — fall back to largest non-empty bin
    for idx in sorted_idx:
        if counts[idx] >= 5:
            bin_pixels = pixels[inverse == idx]
            if is_color:
                return [int(np.median(bin_pixels[:, c])) for c in range(3)]
            else:
                return int(np.median(bin_pixels))

    return None


def _sample_ring_pixels(
    image: np.ndarray,
    fill_mask: np.ndarray,
    ring_width: int,
) -> Optional[np.ndarray]:
    """Extract pixel values from a ring around fill_mask."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1)
    )
    outer = cv2.dilate(fill_mask.astype(np.uint8) * 255, kernel) > 0
    ring = outer & ~fill_mask

    n_samples = int(np.sum(ring))
    if n_samples < 5:
        return None

    if len(image.shape) == 3:
        return image[ring]
    else:
        return image[ring]


def _sample_context_pixels(
    image: np.ndarray,
    fill_mask: np.ndarray,
    region_bbox: Optional[tuple] = None,
    padding_frac: float = 0.12,
) -> Optional[np.ndarray]:
    """Sample pixels from the broader region context (bbox + padding)."""
    if region_bbox is None:
        return None

    x1, y1, x2, y2 = [int(v) for v in region_bbox]
    h_img, w_img = image.shape[:2]

    region_w, region_h = x2 - x1, y2 - y1
    pad = max(10, int(max(region_w, region_h) * padding_frac))
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(w_img, x2 + pad), min(h_img, y2 + pad)

    if bx2 <= bx1 or by2 <= by1:
        return None

    context_mask = np.zeros((h_img, w_img), dtype=bool)
    context_mask[by1:by2, bx1:bx2] = True
    context_mask = context_mask & ~fill_mask

    n_samples = int(np.sum(context_mask))
    if n_samples < 5:
        return None

    if len(image.shape) == 3:
        return image[context_mask]
    else:
        return image[context_mask]


def _desperate_fallback(
    image: np.ndarray,
    region_bbox: Optional[tuple],
    is_color: bool,
) -> list:
    """Last-resort fallback for color sampling."""
    if region_bbox:
        x1, y1, x2, y2 = [int(v) for v in region_bbox]
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 > x1 and y2 > y1:
            context = image[y1:y2, x1:x2]
            if is_color:
                return [int(np.median(context[:, :, c])) for c in range(3)]
            else:
                return int(np.median(context))
    if is_color:
        return [int(np.median(image[:, :, c])) for c in range(3)]
    else:
        return int(np.median(image))


def _feather_mask_edge(
    image: np.ndarray,
    fill_mask: np.ndarray,
    feather_width: int,
) -> np.ndarray:
    """Feather the edge of the filled region to avoid a hard seam."""
    result = image.copy()
    mask_u8 = fill_mask.astype(np.uint8)

    inner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (feather_width * 2 + 1, feather_width * 2 + 1)
    )
    inner = cv2.erode(mask_u8, inner_kernel).astype(bool)
    edge = fill_mask & ~inner
    if not np.any(edge):
        return result

    dist = cv2.distanceTransform(
        (~fill_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    dist = np.clip(dist, 0, feather_width)

    alpha = 1.0 - dist[edge] / feather_width
    alpha = np.clip(alpha, 0, 1)

    edge_blur = cv2.GaussianBlur(image.copy(), (3, 3), 0)

    if len(image.shape) == 3:
        for c in range(3):
            ch = result[:, :, c].astype(np.float64)
            ch[edge] = (
                ch[edge] * alpha
                + edge_blur[:, :, c][edge] * (1.0 - alpha)
            )
            result[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    else:
        ch = result.astype(np.float64)
        ch[edge] = ch[edge] * alpha + edge_blur[edge] * (1.0 - alpha)
        result = np.clip(ch, 0, 255).astype(np.uint8)

    return result
