from __future__ import annotations
from .base import InferenceStrategy
from ..registry import STRATEGIES

@STRATEGIES.register("whole")
class WholeImageStrategy(InferenceStrategy):
    name = "whole"
    def __init__(self, detector):
        super().__init__(detector)
    def run(self, image_bgr):
        return self.detector.detect(image_bgr)
