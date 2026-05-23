from __future__ import annotations
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter

from backend.models.region import TextRegion
from backend.models.render import RenderInfo

logger = logging.getLogger(__name__)


def composite_layers(
    base_image: Image.Image,
    regions: list[TextRegion],
    region_layers: dict[str, Image.Image],
) -> Image.Image:
    result = base_image.copy().convert("RGBA")

    for region in regions:
        layer = region_layers.get(region.id)
        if layer is None:
            continue

        if not region.bbox or len(region.bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in region.bbox]

        if region.render and region.render.overflow:
            logger.warning(f"Region {region.id} has overflow, rendering anyway")

        style = region.style
        opacity = 1.0
        if style:
            opacity = style.opacity

        if opacity < 1.0:
            alpha = layer.split()[3]
            alpha = alpha.point(lambda p: int(p * opacity))
            layer = layer.copy()
            layer.putalpha(alpha)

        layer_rgba = layer.convert("RGBA")

        layer_rgba = _harmonize_layer(base_image, layer_rgba, x1, y1, x2, y2)

        paste_x = x1
        paste_y = y1

        if abs(region.angle) > 0.5:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            layer_w, layer_h = layer_rgba.size
            paste_x = int(cx - layer_w / 2)
            paste_y = int(cy - layer_h / 2)

        temp = Image.new("RGBA", result.size, (0, 0, 0, 0))
        temp.paste(layer_rgba, (paste_x, paste_y), layer_rgba)
        result = Image.alpha_composite(result, temp)

    return result.convert("RGB")


def _harmonize_layer(
    base_image: Image.Image,
    layer: Image.Image,
    x1: int, y1: int, x2: int, y2: int,
) -> Image.Image:
    base_w, base_h = base_image.size
    layer_w, layer_h = layer.size

    if layer_w <= 0 or layer_h <= 0:
        return layer

    pad = 3
    bx1 = max(0, x1 - pad)
    by1 = max(0, y1 - pad)
    bx2 = min(base_w, x2 + pad)
    by2 = min(base_h, y2 + pad)

    if bx2 <= bx1 or by2 <= by1:
        return layer

    base_crop = base_image.crop((bx1, by1, bx2, by2)).convert("RGB")
    base_np = np.array(base_crop)

    border_pixels = []

    h, w = base_np.shape[:2]
    border_size = min(3, h // 2, w // 2)
    if border_size < 1:
        return layer

    top_border = base_np[:border_size, :, :]
    bottom_border = base_np[-border_size:, :, :]
    left_border = base_np[:, :border_size, :]
    right_border = base_np[:, -border_size:, :]

    border_pixels = np.concatenate([
        top_border.reshape(-1, 3),
        bottom_border.reshape(-1, 3),
        left_border.reshape(-1, 3),
        right_border.reshape(-1, 3),
    ])

    if len(border_pixels) == 0:
        return layer

    border_mean = np.mean(border_pixels, axis=0).astype(np.float64)
    border_std = np.std(border_pixels, axis=0).astype(np.float64)
    border_std = np.maximum(border_std, 5.0)

    layer_np = np.array(layer.convert("RGBA"))

    alpha = layer_np[:, :, 3].astype(np.float64) / 255.0
    text_mask = alpha > 0.1

    if not np.any(text_mask):
        return layer

    text_pixels = layer_np[:, :, :3][text_mask].astype(np.float64)
    if len(text_pixels) == 0:
        return layer

    text_mean = np.mean(text_pixels, axis=0)
    text_std = np.std(text_pixels, axis=0)
    text_std = np.maximum(text_std, 5.0)

    local_brightness = _estimate_local_brightness(base_np, h, w)
    target_brightness = np.mean(local_brightness)

    current_brightness = np.mean(text_pixels)
    brightness_ratio = target_brightness / max(current_brightness, 1.0)

    if 0.7 < brightness_ratio < 1.4:
        adjusted = text_pixels * brightness_ratio
    else:
        adjusted = text_pixels

    result_np = layer_np.copy()
    result_np[:, :, :3][text_mask] = np.clip(adjusted, 0, 255).astype(np.uint8)

    noise_level = _estimate_noise_level(base_np)
    if noise_level > 2:
        result_pil = Image.fromarray(result_np)
        result_pil = _add_subtle_noise(result_pil, text_mask, noise_level)
        result_np = np.array(result_pil)

    return Image.fromarray(result_np, "RGBA")


def _estimate_local_brightness(base_np: np.ndarray, h: int, w: int) -> float:
    center_y = h // 2
    center_x = w // 2
    patch_size = min(5, h // 2, w // 2)
    if patch_size < 1:
        return 128.0

    y1 = max(0, center_y - patch_size)
    y2 = min(h, center_y + patch_size)
    x1 = max(0, center_x - patch_size)
    x2 = min(w, center_x + patch_size)

    patch = base_np[y1:y2, x1:x2]
    return float(np.mean(patch))


def _estimate_noise_level(base_np: np.ndarray) -> float:
    if len(base_np.shape) == 3:
        gray = cv2.cvtColor(base_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = base_np.copy()

    h, w = gray.shape
    if h < 4 or w < 4:
        return 0.0

    laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    return float(np.std(laplacian) * 0.1)


def _add_subtle_noise(
    layer: Image.Image,
    text_mask: np.ndarray,
    noise_level: float,
) -> Image.Image:
    result = np.array(layer).astype(np.float64)

    noise = np.random.normal(0, min(noise_level, 8), result[:, :, :3].shape)

    mask_3d = np.stack([text_mask] * 3, axis=-1).astype(np.float64)
    mask_3d = mask_3d * (result[:, :, 3:4] / 255.0)

    result[:, :, :3] += noise * mask_3d * 0.3
    result[:, :, :3] = np.clip(result[:, :, :3], 0, 255)

    return Image.fromarray(result.astype(np.uint8), "RGBA")
