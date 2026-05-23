from __future__ import annotations
from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine


class LamaInpaintEngine:
    def __init__(self, **kwargs):
        self._fallback = OpenCVInpaintEngine()

    def inpaint(self, image, mask, **kwargs):
        return self._fallback.inpaint(image, mask, **kwargs)
