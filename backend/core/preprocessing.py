from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessedScale:
    scale: float
    image: np.ndarray
    gray: np.ndarray
    enhanced: np.ndarray
    sharpened: np.ndarray
    denoised: np.ndarray
    edges: np.ndarray


def preprocess_image(image: np.ndarray, scales: list = None) -> list[PreprocessedScale]:
    if scales is None:
        scales = [1, 2, 3]

    results = []
    for scale in scales:
        resized = _resize(image, scale)
        gray = _to_gray(resized)
        enhanced = _clahe(gray)
        sharpened = _sharpen(enhanced)
        denoised = _denoise(sharpened)
        edges = _canny(denoised)

        results.append(
            PreprocessedScale(
                scale=scale,
                image=resized,
                gray=gray,
                enhanced=enhanced,
                sharpened=sharpened,
                denoised=denoised,
                edges=edges,
            )
        )

    return results


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return image.copy()
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image.copy()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(gray, -1, kernel)


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def _canny(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 50, 150)
