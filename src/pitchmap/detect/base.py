"""The detector interface every detection backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import supervision as sv


@runtime_checkable
class Detector(Protocol):
    """Detects players in a single frame.

    Returning sv.Detections keeps backends interchangeable and lets tiled
    inference wrap any of them without conversion.
    """

    def detect(self, frame: np.ndarray) -> sv.Detections:
        ...
