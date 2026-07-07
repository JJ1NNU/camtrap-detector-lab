"""Detector abstraction. Every model returns a list[Detection] for one BGR image."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]   # xyxy in full-frame pixels
    score: float
    label: str
    class_id: int = -1
    mask: Optional[np.ndarray] = None         # bool HxW at frame size (None if bbox-only)

class Detector(ABC):
    name: str = "detector"
    produces_masks: bool = False

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        """Run the raw model on a single BGR image (a full frame OR a tile)."""

    def describe(self) -> dict:
        return {"name": self.name, "produces_masks": self.produces_masks}
