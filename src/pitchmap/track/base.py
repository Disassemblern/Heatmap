"""The tracker interface every tracking backend implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import supervision as sv


@dataclass(frozen=True)
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


@runtime_checkable
class Tracker(Protocol):
    """Associates detections across frames.

    The frame is passed because appearance-based trackers need pixels for their
    embeddings; motion-only trackers ignore it.
    """

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> tuple[Track, ...]:
        ...
