from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from backend.inpaint_adapters.base import InpaintEngine


class OpenCVInpaintEngine(InpaintEngine):
    def __init__(self, method: str = "telea", radius: int = 3):
        self.method = method
        self.radius = radius

    def inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        method_flag = cv2.INPAINT_TELEA
        if self.method == "ns":
            method_flag = cv2.INPAINT_NS

        radius = kwargs.get("radius", self.radius)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        result = cv2.inpaint(image, mask, inpaintRadius=radius, flags=method_flag)
        return result
