from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PIL import Image


class InpaintEngine(ABC):
    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
