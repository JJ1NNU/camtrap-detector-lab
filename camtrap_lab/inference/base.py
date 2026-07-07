"""Inference strategy: wraps a Detector and decides HOW to apply it (whole / SAHI)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from ..models.base import Detector, Detection

class InferenceStrategy(ABC):
    name: str = "strategy"
    def __init__(self, detector: Detector):
        self.detector = detector
    @abstractmethod
    def run(self, image_bgr: np.ndarray) -> List[Detection]:
        ...
