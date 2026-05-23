from __future__ import annotations
import logging
from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine

logger = logging.getLogger(__name__)


class LamaInpaintEngine:
    """Placeholder — uses OpenCV fallback, not real LaMa.

    Real LaMa / diffusion-based inpainting is not implemented yet.
    Use OpenCVInpaintEngine or SimpleFillEngine directly for now.
    """
    def __init__(self, **kwargs):
        logger.warning("LamaInpaintEngine is a placeholder, falling back to OpenCV")
        self._fallback = OpenCVInpaintEngine()

    def inpaint(self, image, mask, **kwargs):
        return self._fallback.inpaint(image, mask, **kwargs)
